# Official Repository for paper "AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling"
<a href='https://junzhan2000.github.io/AnyGPT.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/pdf/2402.12226.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![](https://img.shields.io/badge/Datasets-AnyInstruct-yellow)](https://huggingface.co/datasets/fnlp/AnyInstruct)

<p align="center">
    <img src="https://raw.githubusercontent.com/OpenMOSS/AnyGPT/main/static/images/logo.png" width="16%"> <br>
</p>

<div align="center">

 | [日本語](docs/README_JP.md) | [English](README.md) |

</div>

## Introduction
We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. The [base model](https://huggingface.co/fnlp/AnyGPT-base) aligns the four modalities, allowing for intermodal conversions between different modalities and text. Furthermore, we constructed the [AnyInstruct](https://huggingface.co/datasets/fnlp/AnyInstruct) dataset based on various generative models, which contains instructions for arbitrary modal interconversion. Trained on this dataset, our [chat model](https://huggingface.co/fnlp/AnyGPT-chat) can engage in free multimodal conversations, where multimodal data can be inserted at will.

AnyGPT proposes a generative training scheme that converts all modal data into a unified discrete representation, using the Next Token Prediction task for unified training on a Large Language Model (LLM). From the perspective of 'compression is intelligence': when the quality of the Tokenizer is high enough, and the perplexity (PPL) of the LLM is low enough, it is possible to compress the vast amount of multimodal data on the internet into the same model, thereby emerging capabilities not present in a pure text-based LLM.
Demos are shown in [project page](https://junzhan2000.github.io/AnyGPT.github.io).

## Example Demonstrations
[![视频标题](http://img.youtube.com/vi/oW3E3pIsaRg/0.jpg)](https://www.youtube.com/watch?v=oW3E3pIsaRg)


## Open-Source Checklist
- [x] Base Model
- [ ] Chat Model
- [x] Inference Code
- [x] Instruction Dataset

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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13_gZPIRG6ShkAbI76-hC_etvfGhry0DZ?usp=sharing)


```bash
python anygpt/src/infer/cli_infer_base_model.py \
--model-name-or-path "path/to/AnyGPT-7B-base" \
--image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
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
  * ```text|image|{caption}```
  * example:
  ```text|image|A bustling medieval market scene with vendors selling exotic goods under colorful tents```
* Image Caption
  * ```image|text|{caption}```
  * example:
  ```image|text|static/infer/image/cat.jpg```
* TTS(random voice)
  * ```text|speech|{speech content}```
  * example:
  ```text|speech|I could be bounded in a nutshell and count myself a king of infinite space.```
* Zero-shot TTS
  * ```text|speech|{speech content}|{voice prompt}```
  * example:
  ```text|speech|I could be bounded in a nutshell and count myself a king of infinite space.|static/infer/speech/voice_prompt1.wav/voice_prompt3.wav```
* ASR
  * ```speech|text|{speech file path}```
  * example: ```speech|text|AnyGPT/static/infer/speech/voice_prompt2.wav```
* Text-to-Music
  * ```text|music|{caption}```
  * example:  
  ```text|music|features an indie rock sound with distinct elements that evoke a dreamy, soothing atmosphere```
* Music Caption
  * ```music|text|{music file path}```
  * example: ```music|text|static/infer/music/features an indie rock sound with distinct element.wav```

**Notes**

For different tasks, we used different language model decoding strategies. The decoding configuration files for image, speech, and music generation are located in ```config/image_generate_config.json```, ```config/speech_generate_config.json```, and ```config/music_generate_config.json```, respectively. The decoding configuration files for other modalities to text are in ```config/text_generate_config.json```. You can directly modify or add parameters to change the decoding strategy.

Due to limitations in data and training resources, the model's generation may still be unstable. You can generate multiple times or try different decoding strategies.

The speech and music response will be saved to ```.wav``` files, and the image response will be saved to a ```jpg```. The filename will be a concatenation of the prompt and the time. The paths to these files will be indicated in the response.


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