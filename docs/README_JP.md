# AnyGPT: 離散シーケンスモデリングを用いた統一マルチモーダル大規模言語モデル

<a href='https://junzhan2000.github.io/AnyGPT.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/pdf/2402.12226.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![](https://img.shields.io/badge/Datasets-AnyInstruct-yellow)](https://huggingface.co/datasets/fnlp/AnyInstruct)

<p align="center">
    <img src="https://raw.githubusercontent.com/OpenMOSS/AnyGPT/main/static/images/logo.png" width="16%"> <br>
</p>

<div align="center">

 | [日本語](README_JP.md) | [English](../README.md) |

</div>


## はじめに
AnyGPTは、音声、テキスト、画像、音楽など様々なモダリティを統一的に処理するための、離散表現を利用した任意のモダリティ間の変換が可能なマルチモーダル言語モデルです。[ベースモデル](https://huggingface.co/fnlp/AnyGPT-base)は4つのモダリティを揃え、異なるモダリティとテキストの間の相互変換を可能にします。さらに、様々な生成モデルを基に、任意のモーダル間変換の指示を含む[AnyInstruct](https://huggingface.co/datasets/fnlp/AnyInstruct)データセットを構築しました。このデータセットで学習された[チャットモデル](https://huggingface.co/fnlp/AnyGPT-chat)は、自由にマルチモーダルデータを挿入できる自由なマルチモーダル会話を行うことができます。

AnyGPTは、全てのモーダルデータを統一された離散表現に変換し、次のトークン予測タスクを用いて大規模言語モデル（LLM）上で統一学習を行う生成学習スキームを提案しています。「圧縮は知性である」という観点から、Tokenizerの品質が十分に高く、LLMのperplexity（PPL）が十分に低ければ、インターネット上の膨大なマルチモーダルデータを同じモデルに圧縮することが可能となり、純粋なテキストベースのLLMにはない能力が現れると考えられます。
デモは[プロジェクトページ](https://junzhan2000.github.io/AnyGPT.github.io)で公開しています。

## デモ例
[![視频标题](http://img.youtube.com/vi/oW3E3pIsaRg/0.jpg)](https://www.youtube.com/watch?v=oW3E3pIsaRg)

## オープンソースチェックリスト
- [x] ベースモデル 
- [ ] チャットモデル
- [x] 推論コード
- [x] 指示データセット

## 推論

### インストール

```bash
git clone https://github.com/OpenMOSS/AnyGPT.git
cd AnyGPT
conda create --name AnyGPT python=3.9
conda activate AnyGPT
pip install -r requirements.txt
```

### モデルの重み
* AnyGPT-baseの重みは[fnlp/AnyGPT-base](https://huggingface.co/fnlp/AnyGPT-base)を確認してください。
* AnyGPT-chatの重みは[fnlp/AnyGPT-chat](https://huggingface.co/fnlp/AnyGPT-chat)を確認してください。 
* SpeechTokenizerとSoundstormの重みは[fnlp/AnyGPT-speech-modules](https://huggingface.co/fnlp/AnyGPT-speech-modules)を確認してください。
* SEED tokenizerの重みは[AILab-CVC/seed-tokenizer-2](https://huggingface.co/AILab-CVC/seed-tokenizer-2)を確認してください。

SpeechTokenizerは音声のトークン化と再構成に使用され、Soundstormはパラ言語情報の補完を担当し、SEED-tokenizerは画像のトークン化に使用されます。

画像の再構成に使用されるunCLIP SD-UNetのモデルの重みと、音楽のトークン化と再構成に使用されるEncodec-32kは自動的にダウンロードされます。

### ベースモデルCLI推論

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

例:
```bash 
python anygpt/src/infer/cli_infer_base_model.py \
--model-name-or-path models/anygpt/base \
--image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
--speech-tokenizer-path models/speechtokenizer/ckpt.dev \
--speech-tokenizer-config models/speechtokenizer/config.json \
--soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt \
--output-dir "infer_output/base"
```

#### 対話
ベースモデルは、テキストから画像、画像キャプション、自動音声認識（ASR）、ゼロショットテキスト音声合成（TTS）、テキストから音楽、音楽キャプションなど、様々なタスクを実行できます。

特定の指示フォーマットに従って推論を行うことができます。

* テキストから画像
  * ```text|image|{キャプション}```
  * 例:
  ```text|image|カラフルなテントの下で異国の商品を売る露店が立ち並ぶ活気あふれる中世の市場の風景```
* 画像キャプション  
  * ```image|text|{キャプション}```
  * 例:
  ```image|text|static/infer/image/cat.jpg```
* TTS（ランダムな声）
  * ```text|speech|{音声の内容}``` 
  * 例:
  ```text|speech|私はナッツの殻の中に閉じ込められていても、無限の空間の王者だと思えます。```
* ゼロショットTTS
  * ```text|speech|{音声の内容}|{声のプロンプト}```
  * 例: 
  ```text|speech|私はナッツの殻の中に閉じ込められていても、無限の空間の王者だと思えます。|static/infer/speech/voice_prompt1.wav/voice_prompt3.wav```
* ASR
  * ```speech|text|{音声ファイルのパス}```
  * 例: ```speech|text|AnyGPT/static/infer/speech/voice_prompt2.wav```  
* テキストから音楽
  * ```text|music|{キャプション}```
  * 例:
  ```text|music|夢のような心地よい雰囲気を醸し出す独特の要素を持つインディーロックサウンド```
* 音楽キャプション
  * ```music|text|{音楽ファイルのパス}```
  * 例: ```music|text|static/infer/music/features an indie rock sound with distinct element.wav```

**注意**

異なるタスクには、異なる言語モデルのデコード戦略を使用しています。画像、音声、音楽生成のデコード設定ファイルは、それぞれ```config/image_generate_config.json```、```config/speech_generate_config.json```、```config/music_generate_config.json```にあります。他のモダリティからテキストへのデコード設定ファイルは、```config/text_generate_config.json```にあります。パラメータを直接変更または追加して、デコード戦略を変更できます。

データと学習リソースの制限により、モデルの生成はまだ不安定な場合があります。複数回生成するか、異なるデコード戦略を試してください。

音声と音楽の応答は ```.wav```ファイルに保存され、画像の応答は```jpg```に保存されます。ファイル名はプロンプトと時間を連結したものになります。これらのファイルへのパスは応答に示されます。

## 謝辞
- [SpeechGPT](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt), [Vicuna](https://github.com/lm-sys/FastChat): 構築したコードベース。
- [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)、[soundstorm-speechtokenizer](https://github.com/ZhangXInFD/soundstorm-speechtokenizer)、[SEED-tokenizer](https://github.com/AILab-CVC/SEED)の素晴らしい仕事に感謝します。

## ライセンス
`AnyGPT`は、[LLaMA2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)の元の[ライセンス](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)の下でリリースされています。

## 引用
AnyGPTとAnyInstructが研究やアプリケーションに役立つと感じた場合は、ぜひ引用してください。
```
@article{zhan2024anygpt,
  title={AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling},
  author={Zhan, Jun and Dai, Junqi and Ye, Jiasheng and Zhou, Yunhua and Zhang, Dong and Liu, Zhigeng and Zhang, Xin and Yuan, Ruibin and Zhang, Ge and Li, Linyang and others},  
  journal={arXiv preprint arXiv:2402.12226},
  year={2024}
}
```