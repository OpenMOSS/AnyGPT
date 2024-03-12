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