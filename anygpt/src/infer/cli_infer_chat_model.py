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
import re
import numpy as np
import traceback
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, EncodecModel, AutoProcessor
from seed2.seed_llama_tokenizer import ImageTokenizer
from PIL import Image
from datetime import datetime
from speechtokenizer import SpeechTokenizer
from m_utils.anything2token import *
from m_utils.read_modality import encode_music_by_path
from m_utils.conversation import get_conv_template
from voice_clone import load_soundstorm, semantic2acoustic
from infer.pre_post_process import extract_content_between_final_tags
from m_utils.prompter import *


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



conversation = get_conv_template('MMGPT')


class AnyGPTChatInference:
    def __init__(
        self, 
        model_name_or_path: str="visual_inter_speech_golden_fs/checkpoint-30000",
        image_tokenizer_path: str="models/seed-tokenizer-2/seed_quantizer.pt",
        output_dir="infer_output/test",
        speech_tokenizer_path:str="models/speechtokenizer/ckpt.dev",
        speech_tokenizer_config:str="models/speechtokenizer/config.json",
        soundstorm_path:str="models/soundstorm/mls_1.pt"
    ):
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
        self.music_segment_duration = 5
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

        #generation
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
        print(content)
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
            
            if prompt_wav.shape[0] == 2:
                prompt_wav = prompt_wav.mean(dim=0).unsqueeze(0)
            prompt_tokens = rearrange(self.speech_tokenizer.encode(prompt_wav.unsqueeze(0)), 'q b n -> b n q')
        else:
            prompt_tokens = None
        # print(prompt_tokens)

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
        task, instruction, 
        image_files=None,
        speech_files=None,
        music_files=None
    ):
        image_list=[]
        music_list=[]
        speech_list=[]
        for image in image_files:
            tokens = self.encode_image(image_path=image.strip())[0]
            processed_inputs = modality_tokens_to_string(tokens=tokens, modality="image")
            # print("image: ", processed_inputs)
            image_list.append(processed_inputs)
        for speech in speech_files:
            tokens = self.encode_speech(speech.strip())
            processed_inputs = modality_tokens_to_string(tokens=tokens, modality="speech")
            # print("speech: ", processed_inputs)
            speech_list.append(processed_inputs)
        for music in music_files:
            tokens = encode_music_by_path(music.strip(), self.music_sample_rate, self.music_tokenizer, self.music_processor, 
                                          self.device, segment_duration=self.music_segment_duration, one_channel=True, start_from_begin=True)
            tokens = tokens[0][0]
            processed_inputs = modality_tokens_to_string(tokens=tokens, modality="music")
            # print("music: ", processed_inputs)
            music_list.append(processed_inputs)
        # 使用sft_prompt
        prompt_seq = self.prompter.generate_insturction_prompt(task,instruction,image_list,speech_list,music_list).strip()
        conversation.append_message(conversation.roles[0], prompt_seq)
        return conversation.get_prompt()

    def postprocess(
        self,
        response: str,
        modality: str,
        input_data: str,
        voice_prompt: str=None
    ):
        # print("post process")
        modality_list = list(modal_special_str.keys())
        for modality in modality_list:
            special_dict = modal_special_str[modality]
            modality_content = extract_content_between_final_tags(response, tag1=special_dict['sos'], tag2=special_dict['eos'])      
            # print("modality_content:",  modality_content)
            if modality_content is None:
                print(f"no {modality}")
                continue
            if modality == "image":
                generated_image = self.decode_image(modality_content)
                now = datetime.now()
                filename = now.strftime("%m%d_%H%M%S") + '.jpg'
                generated_image.save(os.path.join(self.output_dir, input_data[:50]+filename))
                print("image saved: ", os.path.join(self.output_dir, input_data+filename))
            elif modality == "speech":
                generated_wav = self.decode_speech(modality_content, voice_prompt)
                now = datetime.now()
                filename = now.strftime("%m%d_%H%M%S") + '.wav' 
                if voice_prompt:
                    file_name = os.path.join(self.output_dir, input_data[:20] + "--" +
                                            os.path.basename(voice_prompt) + "--" + filename)
                else:
                    file_name = os.path.join(self.output_dir, input_data[:50]+filename)
                print("speech saved: ", file_name)
                torchaudio.save(file_name, generated_wav, self.speech_tokenizer.sample_rate)
            elif modality == "music":
                generated_music = self.decode_music(modality_content)
                now = datetime.now()
                filename = now.strftime("%m%d_%H%M%S") + '.wav'
                file_name = os.path.join(self.output_dir, input_data[:50]+filename)
                print("music saved: ", file_name)
                torchaudio.save(file_name, generated_music, self.music_sample_rate)
            elif modality == "audio":
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
    
    def response(self, task, instruction, to_modality, image_files=None, speech_files=None, music_files=None, voice_prompt=None):
        preprocessed_prompts = (self.preprocess(task, instruction, image_files, speech_files, music_files))
        input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        
        config_path='config/generate_config.json'
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
                
        # print(response)
        if start_of_image in response:
            to_modality = "image"
        elif start_of_speech in response:
            to_modality = "speech"
        elif start_of_music in response:
            to_modality = "music"
        elif start_of_audio in response:
            to_modality = "audio"
        else:
            to_modality = "text"    
        
        # print("extract_content_between_final_tags")
        # 将response写入文本
        with open(os.path.join(self.output_dir, "response.txt"), 'a', encoding='utf-8') as f:
            f.write(response+'\n\n\n')
        try:
            response = extract_content_between_final_tags(response, tag1=f"{chatbot_name}", tag2="<eom>").strip()
        except:
            response = extract_content_between_final_tags(response, tag1=f"{chatbot_name}", tag2="<eos>").strip()
        if to_modality != "text":
            self.postprocess(response, to_modality, instruction, voice_prompt)
        return response
    
    
    def eval_tts(self):
        return

    def forward(
        self, 
        prompts
    ):
        inputs = prompts.split("|")
        task = inputs[0].strip()
        instruction = inputs[1].strip()
        if instruction == "clear":
            conversation.messages=[]
            print("clear conversation history successfully!")
            return
        try:
            to_modality = inputs[2].strip()
        except:
            to_modality = "text"
        try:
            voice_prompt = inputs[3].strip()
        except IndexError:
            voice_prompt = None  
        if voice_prompt=="":
            voice_prompt=None  
        try:
            image_files = inputs[4].strip().split(",")
        except:
            image_files = []
        try:
            speech_files = inputs[5].strip().split(",")
        except:
            speech_files = []
        try:
            music_files = inputs[6].strip().split(",")
        except:
            music_files = []
        if image_files == [""]:
            image_files = []
        if speech_files == [""]:
            speech_files = []
        if music_files == [""]:
            music_files = []
        
        print("task: ", task)
        print("instruction: ", instruction)
        print("to_modality: ", to_modality)
        print("voice_prompt: ", voice_prompt)
        print("image_files: ", image_files)
        print("speech_files: ", speech_files)
        print("music_files: ", music_files)
        if len(speech_files) > 0 and speech_files[0].endswith("jsonl"):
            if instruction == "eval_asr":
                self.eval_asr(speech_files[0])
            elif instruction == "eval_tts": 
                self.eval_tts(speech_files[0])
        else:
            response = self.response(task, instruction,to_modality, image_files, 
                                     speech_files, music_files, voice_prompt)
            print("response:\n", response)
            conversation.append_message(conversation.roles[1], response)
        
    def __call__(self, input):
        return self.forward(input)

    def interact(self):

        prompt = str(input(f"Please talk with AnyGPT chat:\n"))
        while prompt != "quit":
            try:
                self.forward(prompt)
            except Exception as e:
                traceback.print_exc()
                print(e)
            prompt = str(input(f"Please input prompts:\n"))
            
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="output_models/visual_inter_speech_golden_fs/checkpoint-30000")
    parser.add_argument("--image-tokenizer-path", type=str, default="models/seed-tokenizer-2/seed_quantizer.pt")
    parser.add_argument("--speech-tokenizer-path", type=str, default="models/speechtokenizer/ckpt.dev")
    parser.add_argument("--speech-tokenizer-config", type=str, default="models/speechtokenizer/config.json")
    parser.add_argument("--soundstorm-path", type=str, default="models/soundstorm/speechtokenizer_soundstorm_mls.pt")
    
    parser.add_argument("--output-dir", type=str, default="infer_output/test")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    infer = AnyGPTChatInference(
        args.model_name_or_path,
        args.image_tokenizer_path,
        args.output_dir,
        speech_tokenizer_config=args.speech_tokenizer_config,
        speech_tokenizer_path=args.speech_tokenizer_path,
        soundstorm_path=args.soundstorm_path
    )

    infer.interact()