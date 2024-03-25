from pathlib import Path
import re
import os
from shutil import rmtree

from beartype import beartype
from beartype.typing import Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from einops import rearrange

from .dataset import get_dataloader, SoundStormDataset
from .optimizer import get_optimizer
from .soundstorm import SoundStorm
from .tracking import MyTensorBoardTracker

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from speechtokenizer import SpeechTokenizer


# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/soundstorm.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])


class SoundStormTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        model: SoundStorm,
        *,
        num_warmup_steps,
        batch_size,
        train_file_list,
        valid_file_list,
        max_sequence,
        epochs = 20,
        tokenizer: Optional[SpeechTokenizer] = None,
        is_raw_wav: bool = False,
        is_tokens: bool = False,
        tokenizer_kwargs: dict = dict(),
        trainset: Optional[Dataset] = None,
        devset: Optional[Dataset] = None,
        lr = 3e-4,
        initial_lr = 1e-5,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        log_steps = 10,
        save_model_steps = 5000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        split_batches = False,
        drop_last = False,
        num_ckpt_keep = 8,
        num_workers = 8,
        force_clear_prev_results = None
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches = split_batches,
            kwargs_handlers=[ddp_kwargs],
            **accelerate_kwargs
        )

        self.model = model
        # self.tokenizer = tokenizer

        self.register_buffer('steps', torch.Tensor([0]))

        self.epochs = epochs
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        

        # max grad norm

        self.max_grad_norm = max_grad_norm
        
        self.tokenizer = tokenizer
        if exists(self.tokenizer):
            self.tokenizer.eval()
            self.downsample_rate = tokenizer.downsample_rate

        # create dataset
        if exists(trainset):
            self.ds = trainset
        else:
            self.ds = SoundStormDataset(file_list=train_file_list,
                                        is_raw_wav=is_raw_wav,
                                        is_tokens=is_tokens,
                                        tokenizer=self.tokenizer,
                                        max_sequence=int(max_sequence * self.downsample_rate) if is_raw_wav else max_sequence,
                                        **tokenizer_kwargs)
        if exists(devset):
            self.valid_ds = devset
        else:
            self.valid_ds = SoundStormDataset(file_list=valid_file_list,
                                            is_raw_wav=is_raw_wav,
                                            is_tokens=is_tokens,
                                            tokenizer=self.tokenizer,
                                            max_sequence=int(max_sequence * self.downsample_rate) if is_raw_wav else max_sequence,
                                            **tokenizer_kwargs)
        if self.is_main:
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
            
        self.is_raw_wav = is_raw_wav


        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader

        self.dl = get_dataloader(self.ds, is_raw_wav=is_raw_wav, batch_size = batch_size, shuffle = True, drop_last = drop_last, num_workers=num_workers)
        self.valid_dl = get_dataloader(self.valid_ds, is_raw_wav=is_raw_wav, batch_size = batch_size, shuffle = False, drop_last = False, num_workers=num_workers)
        
        # optimizer

        self.optim = get_optimizer(
            model.parameters(),
            lr = lr,
            wd = wd
        )

        # lr and scheduler
        self.lr = lr
        self.initial_lr = initial_lr
        num_train_steps = epochs * self.ds.__len__() // (batch_size * grad_accum_every)
        self.scheduler = CosineAnnealingLR(self.optim, T_max = num_train_steps)
        
        

        
        # prepare with accelerator

        (
            self.model,
            self.tokenizer,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.model,
            self.tokenizer,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.log_steps = log_steps
        self.save_model_steps = save_model_steps
        

        self.results_folder = Path(results_folder)
        self.num_ckpt_keep = num_ckpt_keep

        # if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
        #     rmtree(str(self.results_folder))
        if not self.results_folder.exists():
            self.results_folder.mkdir(parents = True, exist_ok = True)
        
        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": num_warmup_steps, "learning_rate": lr, "initial_learning_rate": lr, "epochs": epochs}
        self.accelerator.init_trackers("soundstorm", config=hps)
        self.best_dev_loss = float('inf')

    def save(self, path, dev_loss):
        if dev_loss < self.best_dev_loss:
            self.best_dev_loss = dev_loss
            torch.save(self.accelerator.get_state_dict(self.model), f'{self.results_folder}/SoundStorm_best_dev.pt')
        ckpts = sorted(Path(path).parent.glob(f'SoundStormTrainer_*'))
        if len(ckpts) > self.num_ckpt_keep:
            [os.remove(c) for c in ckpts[:-self.num_ckpt_keep]]
        pkg = dict(
            model = self.accelerator.get_state_dict(self.model),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict(),
            best_dev_loss = self.best_dev_loss
        )
        torch.save(pkg, path)

    def load(self, path = None, restore_optimizer = True):
        if not exists(path):
            ckpts = sorted(self.results_folder.glob(f'SoundStormTrainer_*'))
            path = str(ckpts[-1])
        model = self.accelerator.unwrap_model(self.model)
        pkg = torch.load(path, map_location='cpu')
        model.load_state_dict(pkg['model'])

        if restore_optimizer:
            self.optim.load_state_dict(pkg['optim'])
            self.scheduler.load_state_dict(pkg['scheduler'])
            if 'best_dev_loss' in pkg.keys():
                self.best_dev_loss = pkg['best_dev_loss']
                if self.is_main:
                    self.print(f'The best dev loss before is {self.best_dev_loss}')

            # + 1 to start from the next step and avoid overwriting the last checkpoint
            self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    def print(self, msg):
        self.accelerator.print(msg)
        
    def tokenize(self, batch):
        
        if not exists(self.tokenizer):
            raise ModuleNotFoundError('No tokenizer in trainer but inputs are raw waves')
        
        wav, length = batch
        if isinstance(length, list):
            length = torch.tensor(length)
        with torch.inference_mode():
            if isinstance(self.tokenizer, torch.nn.parallel.DistributedDataParallel):
                token_ids = self.tokenizer.module.encode(wav.unsqueeze(1))
            else:
                token_ids = self.tokenizer.encode(wav.unsqueeze(1))
        semantic_token_ids = token_ids[0].squeeze()
        acoustic_token_ids = rearrange(token_ids[1:], 'q b n -> b n q')
        mask = torch.ones(semantic_token_ids.shape, dtype=torch.bool, device=self.device)
        length = torch.div(length, self.downsample_rate, rounding_mode='trunc')
        for i in range(semantic_token_ids.size(0)):
            mask[i, length[i]:] = False
        tmp_model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        semantic_token_ids = semantic_token_ids.masked_fill(~mask, tmp_model.semantic_pad_id)
        mask = mask.unsqueeze(-1).repeat(1, 1, tmp_model.num_quantizers)
        acoustic_token_ids = acoustic_token_ids.masked_fill(~mask, tmp_model.pad_id)
        
        return semantic_token_ids, acoustic_token_ids

    # def generate(self, *args, **kwargs):
    #     return self.model.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr

    def train(self):
        
        self.model.train()
        
        grad_accum = 0
        logs = {}
        steps = int(self.steps.item())               
        if steps < self.num_warmup_steps:
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            
        for epoch in range(self.epochs):
            if self.is_main:
                print(f'Epoch:{epoch} start...')
                    
            for batch in self.dl:
                
                if self.is_raw_wav:
                    semantic_token_ids, acoustic_token_ids = self.tokenize(batch)
                else:
                    semantic_token_ids, acoustic_token_ids = batch
                
                loss, acc, _ = self.model(x = acoustic_token_ids,
                                     cond_ids=semantic_token_ids)
                
                accum_log(logs, {'loss': loss.item() / self.grad_accum_every, 'acc': acc.item() / self.grad_accum_every})
                
                self.accelerator.backward(loss / self.grad_accum_every)
                grad_accum += 1
                # self.accelerator.wait_for_everyone()
                
                
                # update params
                if grad_accum == self.grad_accum_every:
                    if exists(self.max_grad_norm):
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()
                    grad_accum = 0
                    
                    # log
                    if self.is_main and not (steps % self.log_steps):
                        self.print(f"Epoch {epoch} -- Step {steps}: loss: {logs['loss']:0.3f}\tacc:{logs['acc']:0.3f}")
                        self.accelerator.log({"train/loss": logs['loss'], "train/acc": logs['acc'], "train/learning_rate": lr}, step=steps)
                    logs = {}
                    
                    self.accelerator.wait_for_everyone()
                    
                    # validate and save model
                    if self.is_main and not(steps % self.save_model_steps):
                        
                        # validate
                        losses = []
                        total_loss = 0.0
                        total_acc = 0.0
                        num = 0
                        self.model.eval()
                        for batch in self.valid_dl:
                            with torch.inference_mode():
                                if self.is_raw_wav:
                                    semantic_token_ids, acoustic_token_ids = self.tokenize(batch)
                                else:
                                    semantic_token_ids, acoustic_token_ids = batch
                                b = semantic_token_ids.size(0)
                                num += b
                                loss, acc, _ = self.model(x = acoustic_token_ids,
                                     cond_ids=semantic_token_ids)
                                total_loss += loss.item() * b
                                losses.append(loss.item())
                                total_acc += acc.item() * b
                        self.print(f'{steps}: valid loss {total_loss / num:0.3f}, valid acc {total_acc / num:0.3f}')  
                        self.accelerator.log({"valid/loss": total_loss / num, "valid/acc": total_acc / num}, step=steps) 
                        
                        # save model
                        model_path = str(self.results_folder / f'SoundStormTrainer_{steps:08d}')
                        self.save(model_path, total_loss / num)                        
                        self.print(f'{steps}: saving model to {str(self.results_folder)}')
                        self.model.train()
                        
                    # Update lr    
                    self.steps += 1
                    steps = int(self.steps.item())               
                    if steps < self.num_warmup_steps:
                        lr = self.warmup(steps)
                        for param_group in self.optim.param_groups:
                            param_group['lr'] = lr
                    else:
                        self.scheduler.step() 
                        lr = self.scheduler.get_last_lr()[0]       
            
        self.print('training complete')
        
    def continue_train(self):
        self.load()
        self.train()