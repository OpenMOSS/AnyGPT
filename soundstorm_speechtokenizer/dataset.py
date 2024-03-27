from torch.utils.data import Dataset, DataLoader
from speechtokenizer import SpeechTokenizer
from einops import rearrange
import torchaudio
import torch
from functools import wraps
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from beartype import beartype
from beartype.typing import Optional

TOKEN_PAD_VALUE = 1024
WAV_PAD_VALUE = 0

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)
        if is_one_data:
            data = tuple(map(lambda x:torch.stack(x), data))
            return data
        outputs = []
        for datum in zip(*data):
            if isinstance(datum[0], torch.Tensor):
                output = fn(datum)
            else:
                output = list(datum)
            outputs.append(output)

        return tuple(outputs)
    return inner

@collate_one_or_multiple_tensors
def tokens_collate_fn(data):
    return pad_sequence(data, batch_first=True, padding_value=TOKEN_PAD_VALUE)

@collate_one_or_multiple_tensors
def wav_collate_fn(data):
    return pad_sequence(data, batch_first=True, padding_value=WAV_PAD_VALUE)

def get_dataloader(ds, is_raw_wav=False, **kwargs):
    collate_fn = wav_collate_fn if is_raw_wav else tokens_collate_fn
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)



class SoundStormDataset(Dataset):
    
    @beartype
    def __init__(self, 
                 file_list: list,
                 is_raw_wav: bool=False,
                 is_tokens: bool=False,
                 tokenizer=None,
                 sample_rate: int= 16000,
                 st_cfg: Optional[str] = None,
                 st_ckpt: Optional[str] = None,
                 max_sequence: int=512,
                 device = 'cpu'):
        self.file_list = file_list
        self.is_raw_wav = is_raw_wav
        self.is_tokens = is_tokens
        self.sample_rate = sample_rate
        if not is_raw_wav and not is_tokens:
            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                self.tokenizer = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt).to(device)
            self.tokenizer.eval()
        self.max_sequence = max_sequence
        self.device = device
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file = self.file_list[index].strip()
        if self.is_tokens:
            tokens = torch.from_numpy(np.load(file))
            if tokens.size(0) > self.max_sequence:
                start = torch.randint(0, tokens.size(0) - self.max_sequence, (1,))
                tokens = tokens[start: (start + self.max_squence)]
            semantic_tokens = tokens[:, 0]
            acoustic_tokens = tokens[:, 1:]
            return semantic_tokens[:self.max_sequence], acoustic_tokens[:self.max_sequence]
        while True:
            try:
                wav, sr = torchaudio.load(file)
                if wav.sum() != 0:
                    break
                raise ValueError('Error audio file')
            except:
                with open('./error_file.txt', 'a+') as f:
                    f.write(file + '\n')
                index -= 1
                file = self.file_list[index].strip()
        if wav.size(0) > 1:
            wav = wav.mean(axis=0)
            wav = wav.unsqueeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if self.is_raw_wav:
            if wav.size(-1) > self.max_sequence:
                start = torch.randint(0, wav.size(-1) - self.max_sequence, (1,))
                wav = wav[:, start: (start + self.max_sequence)]
            return wav.squeeze()[:self.max_sequence], min(wav.size(-1), self.max_sequence)
        wav = wav.to(self.device)
        with torch.inference_mode():
            tokens = self.tokenizer.encode(wav.unsqueeze(0))
        if tokens.size(-1) > self.max_sequence:
            start = torch.randint(0, tokens.size(-1) - self.max_sequence, (1,))
            tokens = tokens[:, :, start:(start + self.max_sequence)]
        semantic_tokens = tokens[0].squeeze()
        acoustic_tokens = rearrange(tokens[1:], 'q b n -> b n q').squeeze()
        return semantic_tokens, acoustic_tokens
    
class Semantic2AcousticDataset(Dataset):
    
    @beartype
    def __init__(self, 
                 file_list: list,
                 audio_root: str,
                 sample_rate: int= 16000,
                 max_sequence: int=512,
                 device = 'cpu'):
        self.audio_root = audio_root
        self.file_list = file_list
        self.sample_rate = sample_rate
        self.max_sequence = max_sequence
        self.device = device
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file = self.file_list[index].strip()
        file_name, units = file.split('\t')
        units = torch.from_numpy(np.array(units.split(' ')).astype(int))
        spk, chapter = file_name.split('_')[:2]
        wav_file = f'{self.audio_root}/{spk}/{chapter}/{file_name}.flac'
        wav, sr = torchaudio.load(wav_file)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(axis=0)
        wav = wav.unsqueeze(0)
        if units.size(-1) > self.max_sequence:
            start = torch.randint(0, units.size(-1) - self.max_sequence, (1,))
            units = units[start:(start + self.max_sequence)]
            wav = wav[:, (start * 320): (start + self.max_sequence)*320]
        return units, wav[:, :self.max_sequence * 320].squeeze(0), min(units.size(-1), self.max_sequence)