# Official Repository for paper "AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling"

<a href="https://junzhan2000.github.io/AnyGPT.github.io/">
  <img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page Badge">
</a>
<a href="https://arxiv.org/pdf/2402.12226.pdf">
  <img src="https://img.shields.io/badge/Paper-Arxiv-red" alt="Paper Arxiv Badge">
</a>
<a href="https://arxiv.org/pdf/2402.12226.pdf">
  <img src="https://img.shields.io/badge/Datasets-AnyInstruct-yellow" alt="Datasets">
</a>


<p align="center">
    <img src="static/images/logo.png" width="16%"> <br>
</p>

## Introduction

We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. The [base model](https://huggingface.co/fnlp/AnyGPT-base) aligns the four modalities, allowing for intermodal conversions between different modalities and text. Furthermore, we constructed the [AnyInstruct](https://huggingface.co/datasets/fnlp/AnyInstruct) dataset based on various generative models, which contains instructions for arbitrary modal interconversion. Trained on this dataset, our [chat model](https://huggingface.co/fnlp/AnyGPT-chat) can engage in free multimodal conversations, where multimodal data can be inserted at will.

AnyGPT proposes a generative training scheme that converts all modal data into a unified discrete representation, using the Next Token Prediction task for unified training on a Large Language Model (LLM). From the perspective of 'compression is intelligence': when the quality of the Tokenizer is high enough, and the perplexity (PPL) of the LLM is low enough, it is possible to compress the vast amount of multimodal data on the internet into the same model, thereby emerging capabilities not present in a pure text-based LLM.
Demos are shown in [project page](https://junzhan2000.github.io/AnyGPT.github.io).

## Example Demonstrations

[![视频标题](http://img.youtube.com/vi/oW3E3pIsaRg/0.jpg)](https://www.youtube.com/watch?v=oW3E3pIsaRg)

## Open-Source Checklist

- [X] Base Model
- [X] Chat Model
- [X] Inference Code
- [X] Instruction Dataset

## Inference

### Installation

```bash
git clone https://github.com/OpenMOSS/AnyGPT.git
cd AnyGPT
conda create --name AnyGPT python=3.9
conda activate AnyGPT
pip install -r requirements.txt
```

### Model Weights

* Check the AnyGPT-base weights in [fnlp/AnyGPT-base](https://huggingface.co/fnlp/AnyGPT-base)
* Check the AnyGPT-chat weights in [fnlp/AnyGPT-chat](https://huggingface.co/fnlp/AnyGPT-chat)
* Check the SpeechTokenizer and Soundstorm weights in [fnlp/AnyGPT-speech-modules](https://huggingface.co/fnlp/AnyGPT-speech-modules)
* Check the SEED tokenizer weights in [AILab-CVC/seed-tokenizer-2](https://huggingface.co/AILab-CVC/seed-tokenizer-2)

The SpeechTokenizer is used for tokenizing and reconstructing speech, Soundstorm is responsible for completing paralinguistic information, and SEED-tokenizer is used for tokenizing images.

The model weights of unCLIP SD-UNet which are used to reconstruct the image, and Encodec-32k which are used to tokenize and reconstruct music will be downloaded automatically.

### Base model CLI Inference

```bash
python anygpt/src/infer/cli_infer_base_model.py \
--model-name-or-path "path/to/AnyGPT-7B-base" \
--image-tokenizer-path 'path/to/model' \
--speech-tokenizer-path "path/to/model" \
--speech-tokenizer-config "path/to/config" \
--soundstorm-path "path/to/model" \
--output-dir "infer_output/base" 
```

for example

```bash
python anygpt/src/infer/cli_infer_base_model.py \
--model-name-or-path models/anygpt/base \
--image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
--speech-tokenizer-path models/speechtokenizer/ckpt.dev \
--speech-tokenizer-config models/speechtokenizer/config.json \
--soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt \
--output-dir "infer_output/base" 
```

#### Interaction

The Base Model can perform various tasks, including text-to-image, image caption, Automatic Speech Recognition (ASR), Zero-shot Text-to-Speech (TTS), Text-to-Music, and Music Captioning.

We can perform inference following a specific instruction format.

* Text-to-Image
  * ``text|image|{caption}``
  * example:
    ``text|image|A bustling medieval market scene with vendors selling exotic goods under colorful tents``
* Image Caption
  * ``image|text|{caption}``
  * example:
    ``image|text|static/infer/image/cat.jpg``
* TTS(random voice)
  * ``text|speech|{speech content}``
  * example:
    ``text|speech|I could be bounded in a nutshell and count myself a king of infinite space.``
* Zero-shot TTS
  * ``text|speech|{speech content}|{voice prompt}``
  * example:
    ``text|speech|I could be bounded in a nutshell and count myself a king of infinite space.|static/infer/speech/voice_prompt3.wav``
* ASR
  * ``speech|text|{speech file path}``
  * example: ``speech|text|AnyGPT/static/infer/speech/voice_prompt2.wav``
* Text-to-Music
  * ``text|music|{caption}``
  * example:
    ``text|music|features an indie rock sound with distinct elements that evoke a dreamy, soothing atmosphere``
* Music Caption
  * ``music|text|{music file path}``
  * example: ``music|text|static/infer/music/features an indie rock sound with distinct element.wav``

**Notes**

For different tasks, we used different language model decoding strategies. The decoding configuration files for image, speech, and music generation are located in ``config/image_generate_config.json``, ``config/speech_generate_config.json``, and ``config/music_generate_config.json``, respectively. The decoding configuration files for other modalities to text are in ``config/text_generate_config.json``. You can directly modify or add parameters to change the decoding strategy.

Due to limitations in data and training resources, the model's generation may still be unstable. You can generate multiple times or try different decoding strategies.

The speech and music response will be saved to ``.wav`` files, and the image response will be saved to a ``jpg``. The filename will be a concatenation of the prompt and the time. The paths to these files will be indicated in the response.

### Chat model CLI Inference

```bash
python anygpt/src/infer/cli_infer_chat_model.py 
\ --model-name-or-path 'path/to/model'
\ --image-tokenizer-path 'path/to/model'
\ --speech-tokenizer-path 'path/to/model'
\ --speech-tokenizer-config 'path/to/config'
\ --soundstorm-path 'path/to/model'
\ --output-dir "infer_output/chat"
```

for example

```bash
python anygpt/src/infer/cli_infer_chat_model.py 
\ --model-name-or-path models/anygpt/chat
\ --image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt 
\ --speech-tokenizer-path models/speechtokenizer/ckpt.dev 
\ --speech-tokenizer-config models/speechtokenizer/config.json 
\ --soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt 
\ --output-dir "infer_output/chat"
```

Instruct format

```bash
interleaved|{text_instruction}|{modality}|{image_path}|{voice_prompt}|{speech_instruction}|{music_path}
```

Where ``text_instruction`` is the input text command, ``speech_instruction`` is the input voice command; only one needs to be specified.

``image_path`` and ``music_path`` are the paths for the input image and music, respectively. ``voice_prompt`` is the specified tone of the model's response; if not specified, a random tone is used.

``modality`` refers to the type of output modality, which can be chosen as speech, image, or music; otherwise, it is considered as text. This will only affect which decoding configuration file under the config directory is used by the model (this is because the model's training is limited, leading to different decoding strategies for different modalities). It can also decode token by token, modifying the decoding strategy to the corresponding modality when generating the start token of the modality.

**example**

* interleaved||image|||static/infer/speech/instruction/Can you draw me a picture of a sunny beach.wav
* interleaved||music|||static/infer/speech/instruction/Give me a similar style of music.wav

To clear the conversation history, please input ``|clear``

### Pretraining and SFT

Please refer to ``scripts/stage1_pretrain.sh`` and ``scripts/stage2_sft.sh``

We provide training data samples for reference. The organization of training formats includes pre-training data in [data/pretrain](https://github.com/OpenMOSS/AnyGPT/tree/main/data/pretrain) and instruction data in [data/instruction](https://github.com/OpenMOSS/AnyGPT/tree/main/data/instruction).
For prompts of different tasks, refer to [task_prompts](https://github.com/OpenMOSS/AnyGPT/blob/16210f829d3b1aa25b0057ebbab0a78057fb59b5/anygpt/src/m_utils/prompter.py#L19), such as plain text dialogue, voice command text reply, text command voice reply, and special prompts for various tasks. You need to process multi-modal data into multi-round dialogue format according to the task template in advance.
We use a voice conversation as an example in the command data, corresponding to the use of task_prompts in the "Speech-Instruction" and "Speech-Response":

```json
[
    {
        "role": "user",
        "message": "<sosp><🗣️1><🗣️1><🗣️1><eosp> Please acknowledge the user's vocal input, create a textual response"
    },
    {
        "role": "assistant",
        "message": "<-Ins-> hello, how are you\n <-Res-> I am fine, thank you <sosp><🗣️2><🗣️2><🗣️2><eosp>"
    }
]
```

## Acknowledgements

- [SpeechGPT](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt), [Vicuna](https://github.com/lm-sys/FastChat): The codebase we built upon.
- We thank the great work from [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer),[soundstorm-speechtokenizer](https://github.com/ZhangXInFD/soundstorm-speechtokenizer), [SEED-tokenizer](https://github.com/AILab-CVC/SEED),

## Lincese

`AnyGPT` is released under the original [License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) of [LLaMA2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf).

## Citation

If you find AnyGPT and AnyInstruct useful in your research or applications, please kindly cite:

```
@article{zhan2024anygpt,
  title={AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling},
  author={Zhan, Jun and Dai, Junqi and Ye, Jiasheng and Zhou, Yunhua and Zhang, Dong and Liu, Zhigeng and Zhang, Xin and Yuan, Ruibin and Zhang, Ge and Li, Linyang and others},
  journal={arXiv preprint arXiv:2402.12226},
  year={2024}
}
```
