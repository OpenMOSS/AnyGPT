import sys
sys.path.append("./")
sys.path.append("./anygpt/src")
import os
import torch
import torchaudio
from einops import rearrange
import argparse
import logging
import json
from tqdm import tqdm
import re
import numpy as np
import traceback
from PIL import Image
from datetime import datetime
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer, GenerationConfig, EncodecModel, AutoProcessor
from seed2.seed_llama_tokenizer import ImageTokenizer
from speechtokenizer import SpeechTokenizer
from m_utils.prompter import *
from m_utils.anything2token import *
from m_utils.read_modality import load_audio, encode_music_by_path
from voice_clone import load_soundstorm, semantic2acoustic
from infer.pre_post_process import extract_text_between_tags

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnyGPTInference:
    def __init__(
        self, 
        model_name_or_path,
        image_tokenizer_path,
        output_dir,
        speech_tokenizer_path,
        speech_tokenizer_config,
        soundstorm_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.prompter = Prompter()
        # image_tokenizer
        print("loading image tokenzier")
        if image_tokenizer_path:
            self.image_tokenizer = ImageTokenizer(model_path=image_tokenizer_path, load_diffusion=True,
                                                  diffusion_model_path="stabilityai/stable-diffusion-2-1-unclip", device=self.device, image_size=224)
        print("loading speech tokenzier")
        self.speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)     
        self.speech_tokenizer.eval()
        self.speech_tokenizer.to(device=self.device)
        self.soundstorm = load_soundstorm(soundstorm_path)
        self.soundstorm.eval()
        self.soundstorm.to(device=self.device)
        print("loading music tokenizer")
        self.music_tokenizer = EncodecModel.from_pretrained("facebook/encodec_32khz")
        self.music_tokenizer.eval()
        self.music_tokenizer.to(device=self.device)
        self.music_processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
        self.music_sample_rate = 32000
        self.music_segment_duration = 20
        print("loading audio tokenizer")
        self.audio_tokenizer = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.audio_tokenizer.eval()
        self.audio_tokenizer.to(device=self.device)
        self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.audio_sample_rate = 24000
        self.audio_segment_duration = 5
        
        # model
        print("loading llm")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )
        self.model.half()  
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 
        self.output_dir = output_dir


    def encode_image(
        self,
        image_path=None,
        image_pil=None,
        image_torch=None
    ):
        assert (image_path is None) + (image_pil is None) + (image_torch is None) == 2
        # need_norm_to_1 = False
        if image_path is not None:
            image_pil = Image.open(image_path).convert('RGB')
        if image_pil is not None:
            image_torch = self.image_tokenizer.processor(image_pil)
            image_torch = image_torch.to(self.device)
        return self.image_tokenizer.encode(image_torch)
    
    
    def decode_image(self, content, negative_indices=None, guidance_scale=10):
        codes = [[int(num) for num in re.findall(r'\d+', content)]]
        indices = torch.Tensor(codes).int().to(self.device)
        if negative_indices is not None:
            negative_indices = negative_indices.to(self.device)
        image = self.image_tokenizer.decode(
            indices,
            negative_indices=negative_indices,
            guidance_scale=guidance_scale,
        )[0]
        return image
     
    def encode_speech(
        self,
        audio_path
    ):
        wav, sr = torchaudio.load(audio_path)
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1, ]
        if sr != self.speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).to(self.device)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = self.speech_tokenizer.encode(wav) # codes: (n_q, B, T)
        return codes[0, 0, :]
    
    def decode_speech(self, content, prompt_path=None):
        if prompt_path:
            # get tokens of prompt
            prompt_wav, sr = torchaudio.load(prompt_path)
            prompt_wav = prompt_wav.to(self.device)
            if sr != self.speech_tokenizer.sample_rate:
                prompt_wav = torchaudio.functional.resample(prompt_wav, sr, self.speech_tokenizer.sample_rate)
            # If it is stereo, take the average to mono
            if prompt_wav.shape[0] == 2:
                prompt_wav = prompt_wav.mean(dim=0).unsqueeze(0)
            prompt_tokens = rearrange(self.speech_tokenizer.encode(prompt_wav.unsqueeze(0)), 'q b n -> b n q')
        else:
            prompt_tokens = None
        # print(prompt_tokens)
        # codes.shape：(1, 1, n)
        semantic_codes = [[int(num) for num in re.findall(r'\d+', content)]]
        # wav: (b, 1, t)
        config_dict = json.load(open('config/generate_config.json', 'r'))
        wav = semantic2acoustic(torch.Tensor(semantic_codes).int().to(self.device), prompt_tokens, 
                                self.soundstorm, self.speech_tokenizer, steps=config_dict['vc_steps'])
        wav = wav.squeeze(0).detach().cpu()
        return wav
    
    def content2rvq_codes(self, content, codebook_size, codebook_num):
        codes = [int(code) for code in re.findall(r'\d+', content)]
        codes = np.array([code % codebook_size for code in codes])
        n = codes.shape[0] // codebook_num
        # Transpose the last two dimensions to match the desired output
        # if can't divide evenly, drop the last few codes
        codes = codes[:n * codebook_num]
        codes = codes.reshape(n, codebook_num).T
        codes = np.expand_dims(codes, 0)
        codes = np.expand_dims(codes, 0)
        codes = torch.tensor(codes).long().to(self.device) 
        return codes
    
    def decode_music(self, content):
        codes = self.content2rvq_codes(content, music_codebook_size, music_codebook_num)
        music = self.music_tokenizer.decode(codes, [None])
        music = music[0].squeeze(0).detach().cpu()
        return music
    
    def decode_audio(self, content):
        codes = self.content2rvq_codes(content, audio_codebook_size, audio_codebook_num)
        audio = self.audio_tokenizer.decode(codes, [None])
        audio = audio[0].squeeze(0).detach().cpu()
        return audio
    
    def preprocess(
        self,
        input_data,
        modality,
        to_modality,
    ):
        # processed_parts = []
        if modality == "text":
            processed_inputs = input_data
        else:
            if modality == "image":
                tokens = self.encode_image(image_path=input_data.strip())[0]
            elif modality == "speech":
                tokens = self.encode_speech(input_data.strip()) # speechtokenizer
            elif modality == "music":
                tokens = encode_music_by_path(input_data.strip(), self.music_sample_rate, self.music_tokenizer, self.music_processor, 
                                              self.device, segment_duration=self.music_segment_duration, one_channel=True, start_from_begin=True)
                tokens = tokens[0][0]
            elif modality == "audio":
                tokens = encode_music_by_path(input_data.strip(), self.audio_sample_rate, self.audio_tokenizer, self.audio_processor, 
                                              self.device, segment_duration=self.audio_segment_duration, one_channel=True, start_from_begin=True)
                tokens = tokens[0][0]
            else:
                raise TypeError("wrong modality")
            processed_inputs = modality_tokens_to_string(tokens=tokens, modality=modality)
        prompt_seq = self.prompter.generate_prompt_input(modality_str=processed_inputs, modality=modality,
                                                         to_modality=to_modality)
        return prompt_seq

    def postprocess(
        self,
        response: str,
        modality: str,
        input_data: str,
        voice_prompt: str=None
    ):
        special_dict = modal_special_str[modality]
        modality_content = extract_text_between_tags(response, tag1=special_dict['sos'], tag2=special_dict['eos'])
        if modality == "image":
            generated_image = self.decode_image(modality_content)
            now = datetime.now()
            filename = now.strftime("%m%d_%H%M%S") + '.jpg'
            generated_image.save(os.path.join(self.output_dir, input_data[:50]+filename))
            print("saved: ", os.path.join(self.output_dir, input_data+filename))
        elif modality == "speech":
            generated_wav = self.decode_speech(modality_content, voice_prompt)
            now = datetime.now()
            filename = now.strftime("%m%d_%H%M%S") + '.wav'  # 设置输出文件名
            if voice_prompt:
                file_name = os.path.join(self.output_dir, input_data[:20] + "--" +
                                         os.path.basename(voice_prompt) + "--" + filename)
            else:
                file_name = os.path.join(self.output_dir, input_data[:50]+filename)
            print("saved: ", file_name)
            torchaudio.save(file_name, generated_wav, self.speech_tokenizer.sample_rate)
        elif modality == "music":
            generated_music = self.decode_music(modality_content)
            now = datetime.now()
            filename = now.strftime("%m%d_%H%M%S") + '.wav'
            file_name = os.path.join(self.output_dir, input_data[:50]+filename)
            print("saved: ", file_name)
            torchaudio.save(file_name, generated_music, self.music_sample_rate)
        elif modality == "audio":
            # 写入modality_content到文本文件
            now = datetime.now()
            os.path.join(self.output_dir, input_data[:50]+now.strftime("%m%d_%H%M%S") + '.txt')
            # with open(os.path.join(self.output_dir, input_data[:50]+now.strftime("%m%d_%H%M%S") + '.txt'), 'w') as f:
            #     f.write(modality_content)
            generated_audio = self.decode_audio(modality_content)
            filename = now.strftime("%m%d_%H%M%S") + '.wav'
            file_name = os.path.join(self.output_dir, input_data[:50]+filename)
            print("saved: ", file_name)
            torchaudio.save(file_name, generated_audio, self.audio_sample_rate)
        return response     
    
    def response(self, modality, to_modality, input_data, voice_prompt=None):
        print(f"modality: {modality}, to_modality: {to_modality}, input_data: {input_data}")
        preprocessed_prompts = (self.preprocess(input_data=input_data, modality=modality,
                                                to_modality=to_modality))
        print("preprocessed_prompts: ", preprocessed_prompts)
        input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        config_path='config/text_generate_config.json'
        if to_modality == "speech":
            config_path='config/speech_generate_config.json'
        elif to_modality == "music":
            config_path='config/music_generate_config.json'
        elif to_modality == "image":
            config_path='config/image_generate_config.json'
        config_dict = json.load(open(config_path, 'r'))
        generation_config = GenerationConfig(    
            **config_dict
        )
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = generated_ids.sequences
        response = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
        print("response: ", response)
        if to_modality == "text":
            response = extract_text_between_tags(response, tag1=f"{chatbot_name} : ", tag2="<eos>").strip()
        else:
            self.postprocess(response, to_modality, input_data, voice_prompt)
        return response
    
    def forward(
        self, 
        prompts
    ):
        inputs = prompts.split("|")
        modality = inputs[0]
        to_modality = inputs[1]
        input_data = inputs[2]
        voice_prompt = None
        if modality == "text" and to_modality == "speech":
            try:
                voice_prompt = inputs[3]
            except IndexError:
                # if not provided, use the random voice prompt
                voice_prompt = None
        response = self.response(modality, to_modality, input_data, voice_prompt)
        print("\nresponse:\n", response)
        
    def __call__(self, input):
        return self.forward(input)

    def interact(self):
        prompt = str(input(f"Please talk with AnyGPT:\n"))
        while prompt != "quit":
            try:
                self.forward(prompt)
            except Exception as e:
                traceback.print_exc()
                print(e)

            prompt = str(input(f"Please input prompts:\n"))
            
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--image-tokenizer-path", type=str, default="models/seed-tokenizer-2/seed_quantizer.pt")
    parser.add_argument("--speech-tokenizer-path", type=str, default="models/speechtokenizer/ckpt.dev")
    parser.add_argument("--speech-tokenizer-config", type=str, default="models/speechtokenizer/config.json")
    parser.add_argument("--soundstorm-path", type=str, default="models/soundstorm/speechtokenizer_soundstorm_mls.pt")
    parser.add_argument("--output-dir", type=str, default="infer_output/base")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    infer = AnyGPTInference(
        model_name_or_path=args.model_name_or_path,
        image_tokenizer_path=args.image_tokenizer_path,
        output_dir=args.output_dir,
        speech_tokenizer_path=args.speech_tokenizer_path,
        speech_tokenizer_config=args.speech_tokenizer_config,
        soundstorm_path=args.soundstorm_path
    )

    infer.interact()