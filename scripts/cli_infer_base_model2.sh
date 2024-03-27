out_dir="infer_output/base" 
mkdir -p ${out_dir}

python anygpt/src/infer/cli_infer_base_model.py \
    --model-name-or-path models/anygpt/base \
    --image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
    --speech-tokenizer-path models/speechtokenizer/ckpt.dev \
    --speech-tokenizer-config models/speechtokenizer/config.json \
    --soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt \
    --output-dir ${out_dir} 



# image|text|data/images/testset/aYQ2uNa.jpg
# image|text|data/images/testset/image-20231121155007517.png
# image|text|data/images/testset/gpt4 test images/4.png


# text|image|a happy dog running on the grass
# text|image|A group of students leaving the school
# text|image|a happy boy playing with his dog
# text|image|a sunset behind a mountain range
# text|image|a beautiful lake, surrounded by mountains
# text|image|a kitten curled up on the ground with its eyes closed behind a tree
# text|image|An animated version of Iron Man
# text|image|A Superman in flight.


# speech|text|data/speech/testset2.jsonl

# text|speech|to be or not to be, this is a question
# text|speech|The primary colors are red, blue, and yellow. These colors are the building blocks of all other colors and are used to create the full spectrum of colors.
# text|speech|Going to the moon is a challenging task that requires a lot of planning and resources. To do this, you will need to develop a spacecraft that can withstand the extreme conditions of the moon's atmosphere
# text|speech|Going to the moon is a challenging task that requires a lot of planning and resources. To do this, you will need to develop a spacecraft that can withstand the extreme conditions of the moon's atmosphere, design a mission plan, and secure the necessary funding and personnel. Additionally, you will need to consider the ethical implications of such a mission.|data/speech/prompt/prompt3.wav
# text|speech|Yes, I do know Stephen Curry.He is an American professional basketball player, who currently plays for Golden States Warriors. He is two-time NBA most valuable player and four-time NBA all star.|data/speech/prompt/prompt3.wav
# text|speech|hello world, hello everyone
# text|speech|hello world
# text|speech|The capital of France is Paris. It is located in the northern part of the country, along the Seine River.
# text|speech|hello world, hello everyone|/mnt/petrelfs/zhanjun.p/mllm/data/speech/prompt/prompt (1).wav
# text|speech|Yes, I do know Stephen Curry.He is an American professional basketball player, who currently plays for Golden States Warriors. He is two-time NBA most valuable player and four-time NBA all star.|/mnt/petrelfs/zhanjun.p/mllm/data/speech/testset/mls-test-1.wav

# text|speech|Going to the moon is a challenging task that requires a lot of planning and resources. To do this, you will need to develop a spacecraft that can withstand the extreme conditions of the moon's atmosphere|/mnt/petrelfs/zhanjun.p/mllm/data/speech/prompt/prompt3.wav
# text|speech|The primary colors are red, blue, and yellow. These colors are the building blocks of all other colors and are used to create the full spectrum of colors.|/mnt/petrelfs/zhanjun.p/mllm/data/speech/prompt/LJ049-0185_24K.wav
# text|speech|The capital of France is Paris. It is located in the northern part of the country, along the Seine River.|/mnt/petrelfs/zhanjun.p/mllm/data/speech/testset/vctk-1.wav
# text|speech|hey guys, i am moss|/mnt/petrelfs/zhanjun.p/mllm/data/speech/prompt/moss-1.wav
# text|speech|hey guys, i am moss. i am an artificial intelligence made by fudan university|/mnt/petrelfs/zhanjun.p/mllm/data/speech/prompt/prompt1.wav
# text|speech|The primary colors are red, blue, and yellow. These colors are the building blocks of all other colors and are used to create the full spectrum of colors.|data/speech/test_case/2.wav

# text|audio|a bird is chirping.
# text|audio|A passionate drum set.
# text|audio|A dog is barking.
# text|audio|A man walking alone on a muddy road.
# text|audio|The roar of a tiger.
# text|audio|A passionate drum set.
# text|audio|The waves crashed against the beach.
# text|audio|A gunshot is being fired.

# audio|text|/mnt/petrelfs/zhanjun.p/mllm/data/audio/沉重的咕噜声..._耳聆网_[声音ID：10492].mp3
# audio|text|/mnt/petrelfs/zhanjun.p/mllm/data/audio/狮子咆哮_耳聆网_[声音ID：11539].wav
# audio|text|/mnt/petrelfs/zhanjun.p/mllm/infer_output/audio_pretrain_4n_2ga_true/checkpoint-37000/a bird is chirping1203_160539.wav
# audio|text|/mnt/petrelfs/zhanjun.p/mllm/infer_output/audio_pretrain_4n_2ga_true/checkpoint-37000/A dog is barking.1203_155916.wav

# text|music|A passionate drum set.
# text|music|a lilting piano melody.
# text|music|Music with a slow and grand rhythm.
# text|music|features an indie rock sound with distinct elements that evoke a dreamy, soothing atmosphere
# text|music|Slow tempo, bass-and-drums-led reggae song. Sustained electric guitar. High-pitched bongos with ringing tones. Vocals are relaxed with a laid-back feel, very expressive.

# sh scripts/infer_cli.sh visual_inter_speech_golden_fs/checkpoint-31000
# sh scripts/infer_cli.sh visual_inter/checkpoint-14000
# sh scripts/infer_cli.sh visual_inter_true/checkpoint-8000
# sh scripts/infer_cli.sh visual_mix_template/checkpoint-5000
# sh scripts/infer_cli.sh speech_pretrain/checkpoint-14000
# sh scripts/infer_cli.sh visual_cc_sbu/checkpoint-4000
# sh scripts/infer_cli.sh visual_laion_no_group/checkpoint-23000
# sh scripts/infer_cli.sh visual_group_4nodes/checkpoint-51000
# sh scripts/infer_cli.sh music_pretrain_4n_4ga/checkpoint-10000
# sh scripts/infer_cli.sh audio_pretrain_4n_2ga/checkpoint-11000

# sh scripts/infer_cli.sh music_pretrain_20s_8n_2ga/checkpoint-58000
# sh scripts/infer_cli.sh audio_pretrain_4n_2ga_true/checkpoint-37000
# sh scripts/infer_cli.sh audio_pretrain_4n_2ga_true/checkpoint-50000