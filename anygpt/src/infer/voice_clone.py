import torch, torchaudio
from soundstorm_speechtokenizer import SoundStorm, ConformerWrapper
from speechtokenizer import SpeechTokenizer
from einops import rearrange
import torch

def load_soundstorm(model_path):
    conformer = ConformerWrapper(codebook_size=1024,
                                num_quantizers=7,
                                conformer={'dim':1024, 
                                        'depth': 12, 
                                        'heads':8, 
                                        'dim_head': 128, 
                                        'attn_flash': False
                                        },
                                    )

    soundstorm = SoundStorm(net=conformer,
                            num_semantic_token_ids=1024,
                            semantic_pad_id=1024,
                            pad_id=1024,
                            schedule = 'cosine')
    soundstorm.load(model_path)
    return soundstorm


def semantic2acoustic(semantic_tokens, prompt_tokens, soundstorm, tokenizer, steps = 1, greedy = True):
    '''
    We aslo support unprompt mode, just let:
    prompt_path = None
    '''
    generated = soundstorm.generate(semantic_tokens=semantic_tokens,
                                    prompt_tokens=prompt_tokens,
                                    steps=steps,
                                    greedy=greedy) 
    wavs = tokenizer.decode(rearrange(generated, 'b n q -> q b n', b=semantic_tokens.size(0))) # wav: (b, 1, t)
    return wavs
