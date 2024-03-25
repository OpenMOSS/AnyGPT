# Official Repository for paper "AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling"
<a href='https://junzhan2000.github.io/AnyGPT.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/pdf/2402.12226.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![](https://img.shields.io/badge/Datasets-AnyInstruct-yellow)](https://huggingface.co/datasets/fnlp/AnyInstruct)

<p align="center">
    <img src="static/images/logo.png" width="16%"> <br>
</p>

## Introduction
We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. 

We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. 

Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model.
Demos are shown in [project page](https://junzhan2000.github.io/AnyGPT.github.io).

## Example Demonstrations
[![视频标题](http://img.youtube.com/vi/oW3E3pIsaRg/0.jpg)](https://www.youtube.com/watch?v=oW3E3pIsaRg)


## Open-Source Checklist
- [ ] Pretraining Model
- [ ] Instruction Model
- [ ] Inference Code
- [x] Instruction Dataset

## Talk with AnyGPT

### Installation

```bash
git clone https://github.com/OpenMOSS/AnyGPT.git
cd AnyGPT
conda create --name AnyGPT python=3.9
conda activate AnyGPT
pip install -r requirements.txt
```

### Download
To talk with SpeechGPT, you should download [SpeechGPT-7B-cm](https://huggingface.co/fnlp/SpeechGPT-7B-cm) and [SpeechGPT-7B-com](https://huggingface.co/fnlp/SpeechGPT-7B-com) locally.

You should download mHuBERT model to ```utils/speech2unit/```. Please see [Speech2unit](https://github.com/0nutation/SpeechGPT/blob/main/speechgpt/utils/speech2unit/README.md) for details.
```bash
s2u_dir="uitls/speech2unit"
cd ${s2u_dir}
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
```

You should download the unit-vocoder to ```utils/vocoder/```. Please see [vocoder](https://github.com/0nutation/SpeechGPT/blob/main/speechgpt/utils/vocoder/README.md) for details.
```bash
vocoder_dir="utils/vocoder/"
cd ${vocoder_dir}
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -O config.json
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -O vocoder.pt
```

### CLI Inference
```bash
python3 speechgpt/src/infer/cli_infer.py \
--model-name-or-path "path/to/SpeechGPT-7B-cm" \
--lora-weights "path/to/SpeechGPT-7B-com" \
--s2u-dir "${s2u_dir}" \
--vocoder-dir "${vocoder_dir} \
--output-dir "output" 
```
**Notes:**
For speech input, you can provide the path to the audio file. For ASR or TTS tasks, you must prefix the speech or text with ```this is input: ```,  otherwise, it may be recognized incorrectly.
The speech response will be saved to a ```.wav``` file, and detailed responses will be saved in a JSON file. The paths to these files will be indicated in the response.

Here are some examples of talking with SpeechGPT:

**Textual dialogue example**


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