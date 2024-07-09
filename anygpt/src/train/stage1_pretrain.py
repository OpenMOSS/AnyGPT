# stage1: visual continue pretrain
import sys
sys.path.append('anygpt/src')

from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
replace_llama_attn_with_flash_attn()


import warnings
warnings.filterwarnings('ignore')
import logging
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import Trainer
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import LlamaForCausalLM, LlamaTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint
import os
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "1000"
from m_utils.prompter import *
from m_utils.anything2token import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

not_twoway_datasets = ['laion2b', 'laion-coco', 'mls']

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/mnt/petrelfs/share_data/llama2_hf/llama-2-7b-hf",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

@dataclass
class DataArguments:
    image_pair_data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."})
    image_inter_data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."})
    speech_data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."})
    mls_data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."})
    music_data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."})
    audio_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."})
    cache_dir: Optional[str] = field(
        default="/mnt/petrelfs/zhanjun.p/mllm/data/both_cache",
        metadata={"help": "Where do you want to store the tokenized data"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "block_size"
            )
        },
    )
    concatenating: bool = field(
        default=True, 
        metadata={"help": "Enable concatenating mode"}
    )
    preprocessing_num_workers: int = field(
        default=50,
        metadata={"help": "preprocessing_num_workers for tokenizing"},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "use_flash_attn"},
    )
    val_set_size: int = field(
        default=1000,
        metadata={"help": "val_set_size"},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "evaluation_strategy"},
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "eval_steps"},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "save_strategy"},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "save_steps"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "num_epochs"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "learning_rate"},
    )
    output_dir: str = field(
        default="",
        metadata={"help": "output_dir"},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    initial_global_step: int = field(
        default=0,
        metadata={"help": "initial_global_step"}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "initial_global_step"}
    )
    only_train_new_embeds: bool = field(
        default=False,
        metadata={"help": "only_train_new_embeds"}
    )
    run_name: str = field(
        default="no run name :)",
        metadata={"help": "run_name"}
    )
    log_on_each_node: bool = field(
        default=False,
        metadata={"help": "log_on_each_node"}
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Log on each process the small summary:
    # 只有主节点logging
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    prompter = Prompter()

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    for token in [user_name, chatbot_name, user_end, chatbot_end]:
        if token not in tokenizer.get_vocab():
            logger.info(f"Add special unit tokens {token} to tokenizer.vocab")
            tokenizer.add_tokens([token])
    
    for modality in modal_special_str.keys():
        prefix=modal_special_str[modality]["prefix"]
        start=modal_special_str[modality]["sos"]
        end=modal_special_str[modality]["eos"]
        modality_vocab_size = modal_special_str[modality]["vocab_size"]
        if start not in tokenizer.get_vocab():
            logger.info(f"Add {modality} tokens <{prefix}0>-<{prefix}{modality_vocab_size-1}> to tokenizer.vocab")
            tokens = [f"<{prefix}{x}>" for x in range(modality_vocab_size)] + [start, end]
            tokenizer.add_tokens(tokens)
        
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path
    )
    # resize embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def generate_template(data_point, x2text_prob=0.5):
        # data_point.keys()有两个值，一个是"text"，取另一个
        modality = None
        for key in data_point.keys():
            if key != "text":
                modality = key
        full_prompt = prompter.generate_template(
            modality_str = data_point[modality],
            text = data_point["text"],
            modality= modality,
            x2text_prob=x2text_prob
        )
        return full_prompt
        
    def tokenize_func(sentence, add_eos_token=True):  
        result = tokenizer(
            sentence,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < tokenizer.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result
    
    def template_and_tokenize(example, add_eos_token=True, x2text_prob=0.5):
        prompt = generate_template(example, x2text_prob)
        return tokenize_func(prompt, add_eos_token)
    
    def tokenize_sent(example, add_eos_token=True):
        return tokenize_func(example["text"], add_eos_token)
    
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        result = {}
        for k in examples.keys():
            v = examples[k]
            if k == "input_ids":
                grouped_input_ids = []
                current_chunk = []
                current_length = 0
                for ids in v:
                    if current_length + len(ids) > block_size:
                        grouped_input_ids.append(current_chunk)
                        current_chunk = []
                        current_length = 0
                    current_chunk.extend(ids)
                    current_length += len(ids)
                # Make sure the last chunk is added
                if current_length > 0:
                    grouped_input_ids.append(current_chunk)
                result[k] = grouped_input_ids
        result["attention_mask"] = [[1 for _ in row] for row in result["input_ids"]]
        result["labels"] = result["input_ids"].copy()
        return result
    
    
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 4096:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 4096
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        
    def load_and_preprocess(data_path, modality, x2text_prob=0.5):
        raw_datasets = load_dataset("json", data_files=data_path)
        # 从后缀名前面抽取出数据集名称
        dataset_name = data_path.split("/")[-1].split(".")[0] 
        print(dataset_name)       
        tokenized_cache_file_names = {
            "train":os.path.join(data_args.cache_dir, modality, dataset_name, 'tokenized', 'train', 'processed_train.arrow'),
            "test":os.path.join(data_args.cache_dir, modality, dataset_name, 'tokenized', 'test', 'processed_valid.arrow'),
        }    
        logger.info(f"loading {dataset_name} tokenized data")
        
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                lambda x: template_and_tokenize(x, x2text_prob),
                batched=False,
                remove_columns=["text", modality],
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
                cache_file_names=tokenized_cache_file_names
        )
        if 'id' in tokenized_datasets.column_names:
            tokenized_datasets = tokenized_datasets.remove_columns(['id'])
        if training_args.val_set_size > 0:
            train_val = tokenized_datasets["train"].train_test_split(
                test_size=training_args.val_set_size, shuffle=True, seed=42
            )
            val_data = train_val["test"]
            tokenized_datasets = tokenized_datasets["train"]
        else:
            val_data = None
            tokenized_datasets = tokenized_datasets["train"]

        group_cache_file_name = os.path.join(data_args.cache_dir, modality, dataset_name, 'group', 'train', 'processed_train.arrow')
        logger.info(f"loading {dataset_name} group data")
        with training_args.main_process_first(desc="grouping tokens together"):
            if data_args.concatenating:
                group_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc=f"block size {block_size}",
                    cache_file_name=group_cache_file_name
                )
            else:
                group_datasets = tokenized_datasets
        train_data = group_datasets
        return train_data, val_data
    
    if data_args.speech_data_path is not None:
        # 按照空格分割出不同的数据集地址
        speech_data_paths = data_args.speech_data_path.split(" ")
        speech_train_datasets = []
        speech_val_datasets = []
        speech_dataset_names = []
        for speech_data_path in speech_data_paths:
            train_data, val_data = load_and_preprocess(speech_data_path, modality="speech")
            speech_train_datasets.append(train_data)
            speech_val_datasets.append(val_data)
            # 抽取出数据集名称
            speech_dataset_names.append(speech_data_path.split("/")[-1].split(".")[0])
        # 训练集使用concatenate_datasets进行拼接
        speech_train_data = concatenate_datasets(speech_train_datasets)
        # 打乱训练集
        speech_train_data = speech_train_data.shuffle(seed=42)
        # 将验证集按名称装入字典
        speech_val_data = {}
        for speech_dataset_name, speech_val_dataset in zip(speech_dataset_names, speech_val_datasets):
            speech_val_data[speech_dataset_name] = speech_val_dataset     

    if data_args.music_data_path is not None:
        music_train_data, music_val_data = load_and_preprocess(data_args.music_data_path, modality="music") 
    
    if data_args.image_pair_data_path is not None:
        # 类似speech数据处理
        image_pair_data_paths = data_args.image_pair_data_path.split(" ")
        image_pair_train_datasets = []
        image_pair_val_datasets = []
        image_pair_dataset_names = []
        for image_pair_data_path in image_pair_data_paths:
            dataset_name = image_pair_data_path.split("/")[-1].split(".")[0]
            x2text_prob = 1.0 if 'caption' in dataset_name else 0
            train_data, val_data = load_and_preprocess(image_pair_data_path, modality="image", x2text_prob=x2text_prob)
            image_pair_train_datasets.append(train_data)
            image_pair_val_datasets.append(val_data)
            image_pair_dataset_names.append(dataset_name)
        image_caption_data_list = []
        image_generation_data_list = []
        image_pair_val_data = {}
        for image_pair_dataset_name, image_pair_train_dataset, image_pair_val_dataset in zip(image_pair_dataset_names, image_pair_train_datasets, image_pair_val_datasets):
            if 'caption' in image_pair_dataset_name:
                image_caption_data_list.append(image_pair_train_dataset)
            else:
                image_generation_data_list.append(image_pair_train_dataset)
            image_pair_val_data[image_pair_dataset_name] = image_pair_val_dataset
        image_caption_train_data = concatenate_datasets(image_caption_data_list)
        image_generation_train_data = concatenate_datasets(image_generation_data_list)

    
    if data_args.mls_data_path is not None:
        mls_train_data, mls_val_data = load_and_preprocess(data_args.mls_data_path, modality="speech")
        
    if data_args.image_inter_data_path is not None:
        dataset_name=data_args.image_inter_data_path.split("/")[-1].split(".")[0]
        image_inter_dataset = load_dataset("text", data_files=data_args.image_inter_data_path)
        image_inter_tokenized_cache_file_names = {
            "train":os.path.join(data_args.cache_dir, 'image', dataset_name, 'tokenized', 'train', 'processed_train.arrow'),
            "test":os.path.join(data_args.cache_dir, 'image', dataset_name, 'tokenized', 'test', 'processed_valid.arrow'),
        }
        logger.info(f"loading image interleaved tokenized data")
        with training_args.main_process_first(desc="image_inter_dataset map tokenization"):
            image_inter_tokenized_datasets = image_inter_dataset.map(
                tokenize_sent,
                batched=False,
                remove_columns=["text"],
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
                cache_file_names=image_inter_tokenized_cache_file_names
            )
        if training_args.val_set_size > 0:
            train_val = image_inter_tokenized_datasets["train"].train_test_split(
                test_size=training_args.val_set_size, shuffle=True, seed=42
            )
            image_inter_val_data = train_val["test"]
            image_inter_tokenized_datasets = image_inter_tokenized_datasets["train"]
        else:
            image_inter_val_data = None
            image_inter_tokenized_datasets = image_inter_tokenized_datasets["train"]
        image_iter_group_cache_file_name = os.path.join(data_args.cache_dir, 'image', dataset_name , 'group', 'train', 'processed_train.arrow')
        logger.info(f"loading image interleaved group data")
        with training_args.main_process_first(desc="grouping texts together"):
            if data_args.concatenating:
                image_inter_group_datasets = image_inter_tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc=f"block size {block_size}",
                    cache_file_name=image_iter_group_cache_file_name
                )
            else:
                image_inter_group_datasets = image_inter_tokenized_datasets
        
    if data_args.audio_data_path is not None:
        audio_data_paths = data_args.audio_data_path.split(" ")
        audio_train_datasets = []
        audio_val_datasets = []
        audio_dataset_names = []
        for audio_data_path in audio_data_paths:
            train_data, val_data = load_and_preprocess(audio_data_path, modality="audio")
            audio_train_datasets.append(train_data)
            audio_val_datasets.append(val_data)
            audio_dataset_names.append(audio_data_path.split("/")[-1].split(".")[0])
        audio_train_data = concatenate_datasets(audio_train_datasets)
        # 打乱训练集
        audio_train_data = audio_train_data.shuffle(seed=42)
        audio_val_data = {}
        for audio_dataset_name, audio_val_dataset in zip(audio_dataset_names, audio_val_datasets):
            audio_val_data[audio_dataset_name] = audio_val_dataset
        


    logger.info(f"interleave_datasets")
    # interleave所有数据集
    if data_args.image_pair_data_path is not None and data_args.speech_data_path is not None and data_args.image_inter_data_path is not None and data_args.mls_data_path is not None and data_args.music_data_path is not None and data_args.audio_data_path is not None:    
        logger.info(f"interleave datasets: speech pair, mls, image pair, mmc4, music, audio")
        # probabilities = [0.23,0.12,0.25,0.1,0.2,0.1] # mls [0.5, 0.5]
        # probabilities = [0.23,0.1,0.22,0.05,0.22,0.18] # mls [0.5, 0.5]
        probabilities = [0.24, 0.10, 0.22, 0.04, 0.25, 0.15]  # mls [0.25, 0.75]
        # probabilities = [0.21, 0.06, 0.22, 0.04, 0.25, 0.22]  # mls [0.25, 0.75]
        # probabilities = [0.23, 0.05, 0.22, 0.04, 0.25, 0.20]  # mls [0.25, 0.75]
        # probabilities = [0.2,0.05,0.22,0.03,0.27,0.23]  # mls [0.25, 0.75]
        logger.info(f"probabilities: {probabilities}")
        train_data = interleave_datasets([speech_train_data, mls_train_data, image_pair_train_data, image_inter_group_datasets, music_train_data, audio_train_data],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # train_data = concatenate_datasets([speech_train_data, mls_train_data, image_pair_train_data, image_inter_group_datasets, music_train_data, audio_train_data])
        # 将验证集字典拆开装进总的验证集
        val_data = {}
        val_data["audio dataset"] = audio_val_data
        val_data["music dataset"] = music_val_data
        for image_pair_dataset_name, image_pair_val_dataset in image_pair_val_data.items():
            val_data[image_pair_dataset_name] = image_pair_val_dataset
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["image interleaved dataset"] = image_inter_val_data
        val_data["mls dataset"] = mls_val_data
    elif data_args.image_pair_data_path is not None and data_args.speech_data_path is not None and data_args.image_inter_data_path is not None and data_args.mls_data_path is not None and data_args.music_data_path is not None:   
        logger.info(f"interleave datasets: speech pair, mls, image pair, mmc4, music")
        probabilities = [0.15, 0.21, 0.3, 0.04, 0.3]  # mls [0.25, 0.75]
        logger.info(f"probabilities: {probabilities}")
        train_data = interleave_datasets([speech_train_data, mls_train_data, image_pair_train_data, image_inter_group_datasets, music_train_data],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # train_data = concatenate_datasets([speech_train_data, mls_train_data, image_pair_train_data, image_inter_group_datasets, music_train_data, audio_train_data])
        # 将验证集字典拆开装进总的验证集
        val_data = {}
        val_data["music dataset"] = music_val_data
        for image_pair_dataset_name, image_pair_val_dataset in image_pair_val_data.items():
            val_data[image_pair_dataset_name] = image_pair_val_dataset
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["image interleaved dataset"] = image_inter_val_data
        val_data["mls dataset"] = mls_val_data
    elif data_args.image_pair_data_path is not None and data_args.speech_data_path is not None and data_args.mls_data_path is not None and data_args.music_data_path is not None:   
        logger.info(f"interleave datasets: speech pair, mls, image pair, music")
        probabilities = [0.20, 0.15, 0.3, 0.3]  # mls [0.25, 0.75]
        logger.info(f"probabilities: {probabilities}")
        train_data = interleave_datasets([speech_train_data, mls_train_data, image_pair_train_data, music_train_data],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # train_data = concatenate_datasets([speech_train_data, mls_train_data, image_pair_train_data, image_inter_group_datasets, music_train_data, audio_train_data])
        # 将验证集字典拆开装进总的验证集
        val_data = {}
        val_data["music dataset"] = music_val_data
        for image_pair_dataset_name, image_pair_val_dataset in image_pair_val_data.items():
            val_data[image_pair_dataset_name] = image_pair_val_dataset
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["mls dataset"] = mls_val_data
    elif data_args.image_pair_data_path is not None and data_args.speech_data_path is not None and data_args.audio_data_path is not None:
        # logger.info(f"interleave datasets: speech pair, image pair, audio")
        probabilities = [0.2, 0.16, 0.34, 0.3] 
        # logger.info(f"probabilities: {probabilities}")
        train_data = interleave_datasets([image_caption_train_data, image_generation_train_data,
                                          speech_train_data, audio_train_data],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # 将验证集字典拆开装进总的验证集
        val_data = {}
        for image_pair_dataset_name, image_pair_val_dataset in image_pair_val_data.items():
            val_data[image_pair_dataset_name] = image_pair_val_dataset
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["audio dataset"] = audio_val_data
    elif data_args.image_pair_data_path is not None and data_args.speech_data_path is not None and data_args.music_data_path is not None:
                # logger.info(f"interleave datasets: speech pair, image pair, audio")
        probabilities = [0.2, 0.22, 0.3, 0.28] 
        # logger.info(f"probabilities: {probabilities}")
        train_data = interleave_datasets([image_caption_train_data, image_generation_train_data,
                                          speech_train_data, music_train_data],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # 将验证集字典拆开装进总的验证集
        val_data = {}
        for image_pair_dataset_name, image_pair_val_dataset in image_pair_val_data.items():
            val_data[image_pair_dataset_name] = image_pair_val_dataset
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["music dataset"] = music_val_data
    elif data_args.music_data_path is not None and data_args.audio_data_path is not None:
        # logger.info(f"interleave datasets: music, audio")
        probabilities = [0.5, 0.5] 
        # logger.info(f"probabilities: {probabilities}")
        train_data = interleave_datasets([music_train_data, audio_train_data],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # 将验证集字典拆开装进总的验证集
        val_data = {}
        val_data["audio dataset"] = audio_val_data
        val_data["music dataset"] = music_val_data
    elif data_args.image_pair_data_path is not None and data_args.speech_data_path is not None and data_args.image_inter_data_path is not None and data_args.mls_data_path is not None:
        # logger.info(f"interleave datasets: speech pair, mls, image pair, mmc4")
        probabilities = [0.3, 0.25, 0.35, 0.1]
        # logger.info(f"probabilities: {probabilities}")
        # 后面可以把mls的比例调高
        train_data = interleave_datasets([speech_train_data, mls_train_data, image_pair_train_data, image_inter_group_datasets],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # 将两个验证集字典拆开装进总的验证集
        val_data = {}
        for image_pair_dataset_name, image_pair_val_dataset in image_pair_val_data.items():
            val_data[image_pair_dataset_name] = image_pair_val_dataset
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["image interleaved dataset"] = image_inter_val_data
        val_data["mls dataset"] = mls_val_data
    elif data_args.image_inter_data_path is not None and data_args.speech_data_path is not None and data_args.mls_data_path is not None:
        probabilities = [0.6, 0.2, 0.2]
        train_data = interleave_datasets([speech_train_data, mls_train_data, image_inter_group_datasets],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # 将两个验证集字典拆开装进总的验证集
        val_data = {}
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["image interleaved dataset"] = image_inter_val_data
        val_data["mls dataset"] = mls_val_data
    elif data_args.image_pair_data_path is not None and data_args.speech_data_path is not None and data_args.mls_data_path is not None:
        probabilities = [0.35, 0.15, 0.5]
        train_data = interleave_datasets([speech_train_data, mls_train_data, image_pair_train_data],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # 将两个验证集字典拆开装进总的验证集
        val_data = {}
        for image_pair_dataset_name, image_pair_val_dataset in image_pair_val_data.items():
            val_data[image_pair_dataset_name] = image_pair_val_dataset
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["mls dataset"] = mls_val_data
    elif data_args.speech_data_path is not None and data_args.mls_data_path is not None:
        # logger.info(f"interleave datasets: speech cv&giga, mls")
        probabilities = [0.7, 0.3]
        train_data = interleave_datasets([speech_train_data, mls_train_data],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # 将两个验证集字典拆开装进总的验证集
        val_data = {}
        for speech_dataset_name, speech_val_dataset in speech_val_data.items():
            val_data[speech_dataset_name] = speech_val_dataset 
        val_data["mls dataset"] = mls_val_data
    elif data_args.image_pair_data_path is not None and data_args.image_inter_data_path:
        # logger.info(f"interleave datasets:image_pair_data, mmc4")
        probabilities = [0.75, 0.25]
        train_data = interleave_datasets([image_pair_train_data, image_inter_group_datasets],
                                        stopping_strategy="all_exhausted", probabilities=probabilities)
        # 将两个验证集字典拆开装进总的验证集
        val_data = {}
        for image_pair_dataset_name, image_pair_val_dataset in image_pair_val_data.items():
            val_data[image_pair_dataset_name] = image_pair_val_dataset
        val_data["image interleaved dataset"] = image_inter_val_data
    elif data_args.image_pair_data_path is not None:
        train_data = image_pair_train_data
        val_data = image_pair_val_data
    elif data_args.music_data_path is not None:
        train_data = music_train_data
        val_data = music_val_data
    elif data_args.audio_data_path is not None:
        train_data = audio_train_data
        val_data = audio_val_data
    else:
        exception_str = "image_pair_data_path and speech_data_path cannot be both None"
        logger.error(exception_str)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    logger.info(f"start training")

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_data if training_args.do_train else None, 
        eval_dataset=val_data if training_args.do_eval else None, 
        data_collator=data_collator
    )

    if training_args.initial_global_step != 0:
        logger.info(f"Set initial global step={training_args.initial_global_step}")
        trainer.state.global_step = training_args.initial_global_step

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()