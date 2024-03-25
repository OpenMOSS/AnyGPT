import math
from random import random, randrange
from functools import wraps
from collections import namedtuple
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, EinMix

from beartype import beartype
from beartype.typing import Union, Dict

from .attend import Attend


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def eval_decorator(fn):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# prob helpers

def sample_prob(prob):
    return random() < prob

def coin_flip():
    return sample_prob(0.5)

# tensor helpers

@beartype
def get_mask_subset_prob(
    mask: Tensor,
    prob: Union[float, Tensor],
    min_mask: int = 0
):
    batch, seq, device = *mask.shape, mask.device

    if isinstance(prob, Tensor):
        prob = rearrange(prob, 'b -> b 1')

    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    # num_padding = (~mask).sum(dim = -1, keepdim = True)
    num_padding = (~mask).sum(dim = -1, keepdim = True) - 1
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# schedules

def linear_schedule(t):
    return 1 - t

def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    return torch.cos(t * math.pi / 2)

# rotary embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# t5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale = 1.,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()

        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, n):
        pos = torch.arange(n, device = self.device).long()
        rel_pos = rearrange(pos, 'j -> 1 j') - rearrange(pos, 'i -> i 1')

        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)

        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale

# conformer

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float32 else 1e-4
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.attend = Attend(
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        rotary_emb = None,
        attn_bias = None
    ):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask = mask, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            Swish(),
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        attn_flash = True,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash)
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None,
        attn_bias = None
    ):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask, rotary_emb = rotary_emb, attn_bias = attn_bias) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

# Conformer

class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        attn_flash = True,
        t5_rel_pos_bias = False
    ):
        super().__init__()

        assert not (t5_rel_pos_bias and attn_flash), 'flash attention is not compatible with learned bias'

        self.dim = dim
        self.layers = nn.ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim_head) if not t5_rel_pos_bias else None
        self.rel_pos_bias = T5RelativePositionBias(dim_head ** 0.5, heads = heads) if t5_rel_pos_bias else None

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout,
                conv_causal = conv_causal,
                attn_flash = attn_flash
            ))

    def forward(self, x, mask = None):
        seq_len = x.shape[-2]

        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None

        for block in self.layers:
            x = block(
                x,
                mask = mask,
                rotary_emb = rotary_emb,
                attn_bias = attn_bias
            )

        return x

# conformer with sum reduction across quantized tokens at the beginning, along with heads

class ConformerWrapper(nn.Module):

    @beartype
    def __init__(
        self,
        *,
        codebook_size,
        num_quantizers,
        conformer: Union[Conformer, Dict[str, any]],
        grouped_quantizers = 1
    ):
        super().__init__()
        self.conformer = conformer

        if isinstance(conformer, dict):
            self.conformer = Conformer(**self.conformer)

        dim = self.conformer.dim

        self.embedding_proj = nn.Sequential(
            nn.Linear(dim * grouped_quantizers, dim),
            nn.LayerNorm(dim)
        ) if grouped_quantizers > 1 else nn.Identity()

        num_codes_with_mask = codebook_size + 1
        num_effective_quantizers = num_quantizers * grouped_quantizers

        self.code_embeds = nn.Embedding(num_codes_with_mask * num_effective_quantizers, dim)

        self.register_buffer('quantizer_offsets', torch.arange(num_effective_quantizers) * num_codes_with_mask, persistent = False)
        self.register_buffer('mask_tokens', self.quantizer_offsets + num_codes_with_mask, persistent = False)

        self.dim = dim
        self.codebook_size = codebook_size

        self.num_codes_with_mask = num_codes_with_mask
        self.num_quantizers = num_quantizers
        self.grouped_quantizers = grouped_quantizers

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * num_effective_quantizers),
            Rearrange('b n (h d) -> b (n h) d', h = num_effective_quantizers)
        )

        # each quantizer codebook would require its own logits weight and bias matrices
        # the amazing einops makes this easy with 'EinMix'

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b (n gq) d -> b n gq d', gq = num_effective_quantizers),
            EinMix(
                'b n gq d -> b n gq l',
                weight_shape = 'gq d l',
                bias_shape = 'gq l',
                gq = num_effective_quantizers,
                l = codebook_size,
                d = dim
            ),
            Rearrange('b ... d -> b (...) d')
        )

    def forward(
        self,
        x,
        *,
        mask = None,
        cond = None,
        sum_embeds = None,
        return_embeddings = False,
        return_logits_and_embeddings = False
    ):
        """
        einops notation:
        b - batch
        n - sequence
        g - groups
        q - quantizers
        d - feature dimension
        """

        n, q, g = x.shape[-1], self.num_quantizers, self.grouped_quantizers
        assert divisible_by(n, g * q), 'sequence must be divisible by number of quantizers'

        x = rearrange(x, 'b (n gq) -> b n gq', gq = g * q)
        x = x + self.quantizer_offsets

        x = self.code_embeds(x)

        x = reduce(x, 'b n (g q) d -> b n (g d)', 'sum', g = g)

        x = self.embedding_proj(x)

        if exists(sum_embeds):
            x = x + sum_embeds

        if exists(cond):
            if cond.ndim == 2:
                cond = rearrange(cond, 'b d -> b 1 d')

            x = x + cond

        x = self.conformer(x, mask = mask)
        embeds = self.heads(x)

        if return_embeddings or not exists(self.to_logits):
            return embeds

        logits = self.to_logits(embeds)

        if return_logits_and_embeddings:
            return logits, embeds

        return logits

# for main logits as well as self token critic

class LogitHead(nn.Module):
    def __init__(
        self,
        net: ConformerWrapper,
        logit_dim
    ):
        super().__init__()
        self.net = net
        dim = net.dim
        self.to_logits = nn.Linear(dim, logit_dim)

    def forward(self, x):
        embed = self.net(x, return_embeddings = True)
        return self.to_logits(embed)

# main soundstorm class, which is just a maskgit

LossBreakdown = namedtuple('LossBreakdown', ['generator_loss', 'critic_loss'])

class SoundStorm(nn.Module):
    
    def __init__(self,
                 net: ConformerWrapper,
                 num_semantic_token_ids,
                 semantic_pad_id = -1,
                 pad_id = None,
                 schedule = 'linear',
                 ):
        super().__init__()
        self.net = net
        self.dim = net.dim
        self.num_tokens = net.codebook_size
        self.pad_id = pad_id
        self.num_semantic_token_ids = num_semantic_token_ids
        self.semantic_token_emb = nn.Embedding(num_semantic_token_ids, self.dim)
        self.semantic_pad_id = semantic_pad_id
        if callable(schedule):
            self.schedule_fn = schedule
        elif schedule == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule}')
        self.num_quantizers = net.num_quantizers
        self.mask_id = net.codebook_size
    
    def get_condition(self, token_ids, length=None):
        mask = token_ids != self.semantic_pad_id
        token_ids = token_ids.masked_fill(~mask, 0)
        semantic_tokens = self.semantic_token_emb(token_ids)
        cond_tokens = semantic_tokens.masked_fill(~rearrange(mask, '... -> ... 1'), 0.)
        if exists(length):
            cond_length = cond_tokens.size(-2)
            if cond_length < length:
                cond_length = F.pad(cond_tokens, (0, 0, 0, length - cond_length), value=0.)
            elif cond_length > length:
                cond_tokens = cond_tokens[:, :length]
        return cond_tokens
    
    @property
    def device(self):
        return next(self.net.parameters()).device
    
    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()
        params = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(params, strict = strict)
        
    @torch.no_grad()
    @eval_decorator
    def generate(self,
                  semantic_tokens,
                  prompt_tokens = None,
                  steps = 8,
                  num_full_sampling_levels = 1,
                  topk_pres = 0.7,
                  greedy=True,
                  ):
        device = self.device
        batch_size, seq_length = semantic_tokens.shape
        mask = torch.ones((batch_size, semantic_tokens.size(-1), self.num_quantizers), device=device)
        masked = mask * self.mask_id
        cond_tokens = semantic_tokens
        
        if exists(prompt_tokens):
            prompt_semantic_tokens = prompt_tokens[:, :, 0]
            prompt_acoustic_tokens = prompt_tokens[:, :, 1:]
            cond_tokens = torch.cat([prompt_semantic_tokens, cond_tokens], axis=-1)
            masked = torch.cat([prompt_acoustic_tokens, masked], axis=1)
            mask = torch.cat([torch.zeros_like(prompt_acoustic_tokens, dtype=torch.bool, device=device), mask.bool()], axis=1)
        else:
            mask = mask.bool()
            
            
        prompt_mask = mask.clone()
        seq_mask = torch.ones_like(cond_tokens, dtype=torch.bool, device=device)
        cond_tokens = self.semantic_token_emb(cond_tokens)
        seq_mask_with_quantizer = repeat(seq_mask, 'b n -> b (n q)', q = self.num_quantizers)
        times = torch.linspace(0., 1., steps + 1, device=device)
        rand_mask_probs = cosine_schedule(times)
        rand_mask_probs = rearrange(rand_mask_probs, 'n -> n 1')
        seq_len_from_mask = reduce(seq_mask, 'b n -> b', 'sum')
        all_mask_num_tokens = (rand_mask_probs * seq_len_from_mask).long()
        
        for q in range(self.num_quantizers):
            all_mask_num_tokens = all_mask_num_tokens if q < num_full_sampling_levels else torch.zeros((1, batch_size), dtype = torch.long, device = device)
            for i, mask_num_tokens in enumerate(all_mask_num_tokens):
                masked_input = rearrange(masked, 'b n q -> b (n q)')
                logits = self.net(masked_input.long(), mask=seq_mask, cond=cond_tokens)
                if greedy and (q >= num_full_sampling_levels or (mask_num_tokens == 0).all()):
                    sampled_ids = logits.argmax(axis=-1)
                else:
                    logits = top_k(logits, thres=topk_pres)
                    temperature = 1.0 * i / steps
                    sampled_ids = gumbel_sample(logits, temperature=temperature)
                if q >= num_full_sampling_levels:
                    masked[:, :, q:] = rearrange(sampled_ids, 'b (n q) -> b n q', q=self.num_quantizers)[:, :, q:]
                    mask[:, :, q] = False
                    continue
                
                masked = torch.where(mask, rearrange(sampled_ids, 'b (n q) -> b n q', q=self.num_quantizers), masked)
                if (mask_num_tokens == 0).all():    
                    continue
                
                scores = 1 - logits.softmax(dim=-1)
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')
                mask = torch.zeros_like(scores, dtype = torch.bool, device=device)
                mask_value = -torch.finfo(scores.dtype).max
                scores = scores.masked_fill(~seq_mask_with_quantizer, mask_value)
                scores_sorted = scores.argsort(dim = -1, descending = True)
                mask_num_tokens = rearrange(mask_num_tokens, 'b -> b 1')
                mask_tokens = scores_sorted[:, :mask_num_tokens]
                rows = torch.arange(mask_tokens.size(0)).unsqueeze(-1).expand_as(mask_tokens)
                mask[rows, mask_tokens] = True
                mask = rearrange(mask, 'b (n q) -> b n q', q = self.num_quantizers)
                mask[:, :, (q + 1):] = True
                mask = mask & prompt_mask
                
                masked = masked.masked_fill(mask, self.mask_id)
        output = torch.cat([semantic_tokens.unsqueeze(-1), masked[:, -seq_length:]], axis=-1)
        return output.detach().long()
        
        
    def forward(self,
                x,
                cond_ids,
                mask = None,
                generator_sample_temperature = 1,
                **kwargs):
        b, n, q = x.shape
        device = x.device
        seq_mask = mask
        
        if not exists(seq_mask):
            seq_mask = torch.ones((b, n), device=device, dtype=torch.bool)
        
        if exists(self.pad_id):
            pad_mask = (x == self.pad_id).any(dim = -1)
            seq_mask = seq_mask & ~pad_mask
            
        cond_tokens = self.get_condition(cond_ids)
        orig_seq = rearrange(x.clone(), 'b n q -> b (n q)')
        
        min_seq_len = seq_mask.sum(dim =-1).amin()
        
        # sample prompt delimiter
        t = randrange(0, min_seq_len - 1)
        
        mask = seq_mask[:, t:] 
        
        # sample time position mask
        
        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        rand_probs = self.schedule_fn(rand_times)
        
        mask = get_mask_subset_prob(mask, rand_probs)  
        
        # random quantizer position
        
        q = randrange(0, self.num_quantizers)
        
        masked = torch.where(mask, self.mask_id, x[:, t:, q])
        masked = rearrange(torch.cat((x[:, :t, q], masked), dim=1), 'b n -> b n 1')
        masked = torch.cat((x[:, :, :q], masked, x[:, :, q + 1:]), dim=2)
        masked[:, t:, q + 1:] = self.mask_id
        masked = rearrange(masked, 'b n q -> b (n q)')
        
        prompt_mask = torch.full((b, t), False, device=device)
        lower_quantizers_mask = torch.full((b, n, q), False, device=device)
        upper_quantizers_mask = torch.full((b, n, self.num_quantizers - q - 1), True, device=device)
        
        # upper_quantizers_mask in prompt also should be False

        upper_quantizers_mask[:, :t, :] = False
        mask = rearrange(torch.cat((prompt_mask, mask), dim=1), 'b n -> b n 1')
        mask = torch.cat((lower_quantizers_mask, mask, upper_quantizers_mask), dim = 2)
        
        # above is the right mask, but when computing loss, only consider level q
        
        mask[:, :, q + 1:] = False
        mask = rearrange(mask, 'b n q -> b (n q)')
        
        logits = self.net(masked,
                          mask=seq_mask,
                          cond = cond_tokens,
                          **kwargs)
        
        # CE loss
        loss = F.cross_entropy(logits[mask],
                               orig_seq[mask])
        
        sampled_ids = gumbel_sample(logits, temperature = default(generator_sample_temperature, random()))
        acc = (sampled_ids[mask] == orig_seq[mask]).sum() / mask.sum()
        generated = torch.where(mask, sampled_ids, orig_seq)
        
        return loss, acc, generated
        
            