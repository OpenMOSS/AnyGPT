# Running AnyGPT with Docker

This README explains how to run AnyGPT using Docker.

## Prerequisites

- Docker is installed
- NVIDIA Container Toolkit is installed if running in a GPU environment

## Steps

1. Build the Docker image by running the following command:
   ```bash
   docker-compose up --build
   ```

2. Download the models:
   ```bash
   docker-compose run anygpt python /app/scripts/download_models.py
   ```

3. Run the inference:
   ```bash
   docker-compose run anygpt python anygpt/src/infer/cli_infer_base_model.py \
     --model-name-or-path models/anygpt/base \
     --image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
     --speech-tokenizer-path models/speechtokenizer/ckpt.dev \
     --speech-tokenizer-config models/speechtokenizer/config.json \
     --soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt \
     --output-dir "infer_output/base"
   ```

4. The inference results will be output to the `docker/infer_output/base` directory.

## Troubleshooting

- If the model download fails, check the `download_models.py` script and update the URLs if necessary.
- If the inference execution fails, check the command arguments and ensure that the model paths are correct.

## Notes

- Downloading the models and running the inference requires a large amount of memory and disk space. Ensure that sufficient resources are available.