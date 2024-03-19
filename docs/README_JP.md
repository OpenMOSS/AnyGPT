# 論文「AnyGPT: 離散シーケンスモデリングを用いた統一マルチモーダル大規模言語モデル」の公式リポジトリ

<a href='https://junzhan2000.github.io/AnyGPT.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/pdf/2402.12226.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![](https://img.shields.io/badge/Datasets-AnyInstruct-yellow)](https://huggingface.co/datasets/fnlp/AnyInstruct)

<p align="center">
    <img src="https://raw.githubusercontent.com/OpenMOSS/AnyGPT/main/static/images/logo.png" width="16%"> <br>
</p>

<div align="center">

 | [日本語](README_JP.md) | [English](../README.md) |

</div>

## はじめに (Introduction)
AnyGPTは、音声、テキスト、画像、音楽など、様々なモダリティを統一的に処理するために、離散表現を利用したany-to-anyマルチモーダル言語モデルです。AnyGPTは、現在の大規模言語モデル（LLM）のアーキテクチャやトレーニングパラダイムを変更することなく、安定してトレーニングすることができます。代わりに、データレベルの前処理のみに依存しているため、新しい言語を組み込むのと同じように、LLMに新しいモダリティをシームレスに統合することができます。

マルチモーダルアラインメントの事前トレーニング（multimodal alignment pre-training）のために、マルチモーダルなテキスト中心のデータセットを構築しました。生成モデルを利用して、大規模なany-to-anyマルチモーダル指示データセット（any-to-any multimodal instruction dataset）を初めて合成しました。このデータセットは、様々なモダリティが複雑に絡み合った108kサンプルのマルチターン会話で構成されており、モデルが任意のマルチモーダル入力と出力の組み合わせを扱えるようになっています。

実験結果から、AnyGPTはany-to-anyマルチモーダル会話を可能にし、すべてのモダリティにおいて専門モデルに匹敵するパフォーマンスを達成することが示されました。これにより、離散表現が言語モデル内の複数のモダリティを効果的かつ便利に統一できることが証明されました。
デモは[プロジェクトページ](https://junzhan2000.github.io/AnyGPT.github.io)で公開されています。

## デモ例 (Example Demonstrations)
[![視频标题](http://img.youtube.com/vi/oW3E3pIsaRg/0.jpg)](https://www.youtube.com/watch?v=oW3E3pIsaRg)

## オープンソースチェックリスト (Open-Source Checklist)
- [ ] 事前学習モデル (Pretraining Model)
- [ ] 指示モデル (Instruction Model)
- [ ] 推論コード (Inference Code)
- [x] 指示データセット (Instruction Dataset)

## 引用 (Citation)
AnyGPTとAnyInstructが研究やアプリケーションに役立つと感じた場合は、以下のように引用してください。
```
@article{zhan2024anygpt,
  title={AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling},
  author={Zhan, Jun and Dai, Junqi and Ye, Jiasheng and Zhou, Yunhua and Zhang, Dong and Liu, Zhigeng and Zhang, Xin and Yuan, Ruibin and Zhang, Ge and Li, Linyang and others},
  journal={arXiv preprint arXiv:2402.12226},
  year={2024}
}
```