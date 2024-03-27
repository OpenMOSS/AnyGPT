FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        git \
        gcc \
        curl \
        wget \
        sudo \
        pciutils \
        python3-all-dev \
        python-is-python3 \
        python3-pip \
        ffmpeg \
        libsdl2-dev \
        pulseaudio \
        alsa-utils \
        portaudio19-dev \
    && pip install pip -U

WORKDIR /app

RUN git clone https://github.com/Sunwood-ai-labs/AnyGPT-JP.git .

RUN pip install -r requirements.txt

RUN mkdir -p models/anygpt \
    && mkdir -p models/seed-tokenizer-2 \
    && mkdir -p models/speechtokenizer \
    && mkdir -p models/soundstorm

RUN pip install bigmodelvis \
    && pip install peft \
    && pip install --upgrade huggingface_hub

# COPY scripts/download_models.py /app/scripts/download_models.py
# RUN python scripts/download_models.py
