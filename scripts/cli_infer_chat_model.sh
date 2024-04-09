ckpt=$1
model_name=$2
ckpt_name="${ckpt#output_models/}"
echo ${ckpt_name}
out_dir="./infer_output/${ckpt_name}" 
mkdir -p ${out_dir}
echo "output dir: ${out_dir}"


python ./anygpt/src/infer/cli_infer_chat_model.py \
    --model-name-or-path /mnt/petrelfs/zhanjun.p/mllm/${ckpt} \
    --image-tokenizer-path "models/seed-tokenizer-2/seed_quantizer.pt" \
    --output-dir ${out_dir}

# srun -p llm_h -n1 --ntasks-per-node 1 --gres=gpu:1 --job-name infer --kill-on-bad-exit=1 --quotatype=reserved sh scripts/infer_visual_sft.sh output_models/save_sft_visual_no_g/20k

# Image Conversation|Where does your finger point on the map?|text||data/images/testset/aus.jpeg
# Image Conversation|Where does your finger point on the map?|text||data/images/testset/cn2.jpg
# Image Conversation|Who's the guy in the picture?|text||data/evaluation/image/test/ironman.png
# Text-to-Image Generation|


# Text-to-Music Generation|Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach|music
# customized|generate a music, about Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach|music
# Image Conversation|what is this|text||data/images/testset/au1.jpg
# Image Conversation|What is funny about this image? Describe it panel by panel|text||data/images/testset/gpt4 test images/1.png
# Text-to-Image Generation|An animated version of Iron Man.|image
# Text-to-Image Generation|A stick figure version of Iron Man.|image
# Text-to-Image Generation|Pencil drawing of Iron Man.|image
# Text-to-Image Generation|Iron Man sketch.|image
# Text-to-Image Generation|Animated Iron Man.|image
# Text-to-Image Generation|draw a cute dog
# Text-to-Image Generation|draw a yellow car
# |Can you generate an image of a cartoon cat in a garden?|image
# |It looks so cute! Let's name the cat Cookie. Can you make Cookie read a book?|text
# |It looks like Cookie enjoys reading. Now let's make Cookie cook food in the kitchen.|text
# |Cookie looks like an excellent chef. It is time for Cookie to practice playing the piano.|text
# |I love these paintings! Can you write a story about Cookie?|text

# interleaved|Can you show me a picture of a sunny beach?|image
# interleaved|Can you give me a similar style of music?|music

# interleaved|Can you show me a picture of winter?|image
# interleaved|What kind of music do you associate with it?|music
# interleaved|Can you translate the emotion in this picture into music?|music||data/mmichat_sample/jiqing.png
# interleaved|Can you translate the emotion in this picture into music?|music||data/mmichat_sample/ningjing.png
# interleaved|Can you show the content of music as an image?|image||||data/mmichat_sample/0a1af329_0.wav
# interleaved|Can you translate the emotion in this picture into music?|music||infer_output/sft_mmi1/checkpoint-2000/xiaoqiao.jpeg

# interleaved|What emotion do you hear in this melody?|text||||infer_output/music_pretrain_4n_4ga/checkpoint-37000/A passionate drum set1130_161953.wav

# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/evaluation/speech/question/what's your name-en-US-JennyNeural.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/evaluation/speech/question/Can you draw me a picture of a sunny beach.wav


# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_0.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_1.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_2.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_3.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_4.wav


# Speech-Instruction||speech|data/evaluation/speech/prompt/emov_db_anger_pt.wav||data/speech/test_conv/audio_0.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_1.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_2.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_3.wav
# Speech-Instruction||speech|data/speech/prompt/prompt3.wav||data/speech/test_conv/audio_4.wav


# Speech-Instruction|None|speech|data/speech/test_conv/question_9.wav
# Speech-Instruction|None|speech|data/speech/test_conv/question_13.wav

# Speech-Instruction2|None|speech|data/speech/test_conv/audio_0.wav
# Speech-Instruction2|None|speech|data/speech/test_conv/audio_1.wav
# Speech-Instruction2|None|speech|data/speech/test_conv/audio_2.wav
# Speech-Instruction2|None|speech|data/speech/test_conv/audio_3.wav
# Speech-Instruction2|None|speech|data/speech/test_conv/audio_4.wav
# Speech-Instruction2|None|speech|data/speech/test_conv/question_9.wav
# Speech-Instruction2|None|speech|data/speech/test_conv/question_13.wav

# Text-Instruction|what are primary colors?|text
# Text-Instruction|what are primary colors?|speech
# Text-Instruction2|what are primary colors?|speech




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
# text|speech|Going to the moon is a challenging task that requires a lot of planning and resources. To do this, you will need to develop a spacecraft that can withstand the extreme conditions of the moon's atmosphere|
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
