python anygpt/src/infer/cli_infer_chat_model.py 
    \ --model-name-or-path /mnt/petrelfs/zhanjun.p/mllm/output_models/sft_mmgpt_mmi_speech2/checkpoint-4500
    \ --image-tokenizer-path /mnt/petrelfs/zhanjun.p/mllm/AnyGPT/models/seed-tokenizer-2/seed_quantizer.pt 
    \ --speech-tokenizer-path /mnt/petrelfs/zhanjun.p/mllm/AnyGPT/models/speechtokenizer/ckpt.dev 
    \ --speech-tokenizer-config /mnt/petrelfs/zhanjun.p/mllm/AnyGPT/models/speechtokenizer/config.json 
    \ --soundstorm-path /mnt/petrelfs/zhanjun.p/mllm/AnyGPT/models/soundstorm/speechtokenizer_soundstorm_mls.pt 
    \ --output-dir "infer_output/chat"