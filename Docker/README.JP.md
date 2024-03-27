# AnyGPTのDockerでの実行方法

このREADMEでは、AnyGPTをDockerを用いて実行する方法を説明します。

## 前提条件

- Dockerがインストール済みであること
- GPU環境で実行する場合は、NVIDIA Container Toolkitがインストール済みであること

## 手順


1. 以下のコマンドを実行して、Dockerイメージをビルドします。
   ```bash
   docker-compose up --build
   ```

2. モデルをダウンロードします。
   ```bash
   docker-compose run anygpt python /app/scripts/download_models.py
   ```

3. 推論を実行します。
   ```bash
   docker-compose run anygpt python anygpt/src/infer/cli_infer_base_model.py \
     --model-name-or-path models/anygpt/base \
     --image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
     --speech-tokenizer-path models/speechtokenizer/ckpt.dev \
     --speech-tokenizer-config models/speechtokenizer/config.json \
     --soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt \
     --output-dir "infer_output/base"
   ```

6. 推論結果は `docker/infer_output/base` ディレクトリに出力されます。

## トラブルシューティング

- モデルのダウンロードに失敗する場合は、`download_models.py`スクリプトを確認し、必要に応じてURLを更新してください。
- 推論の実行に失敗する場合は、コマンドの引数を確認し、モデルのパスが正しいことを確認してください。

## 注意事項

- モデルのダウンロードと推論の実行には、大量のメモリとディスク容量が必要です。十分なリソースを確保してください