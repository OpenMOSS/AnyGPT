from huggingface_hub import snapshot_download

def download_models():
    snapshot_download(repo_id='fnlp/AnyGPT-base', local_dir='models/anygpt/base')
    snapshot_download(repo_id='AILab-CVC/seed-tokenizer-2', local_dir='models/seed-tokenizer-2')
    snapshot_download(repo_id='fnlp/AnyGPT-speech-modules', local_dir='models')

if __name__ == '__main__':
    download_models()