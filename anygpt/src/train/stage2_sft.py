# stage1: visual continue pretrain
import sys
sys.path.append('anygpt/src')
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
replace_llama_attn_with_flash_attn()
import os
import json
import copy
import warnings
warnings.filterwarnings('ignore')
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from torch.utils.data import Dataset
import torch
import transformers
from transformers import Trainer
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import LlamaForCausalLM, LlamaTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "1000"
from m_utils.prompter import *
from m_utils.anything2token import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

seed=42
IGNORE_TOKEN_ID=-100

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="share_data/llama2_hf/llama-2-7b-hf",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

@dataclass
class DataArguments:
    mmi_datasets: str = field(
        default=None,
        metadata={"help": "mmi_datasets"},
    )
    speech_conv_datasets: str = field(
        default=None,
        metadata={"help": "speech_conv_datasets"},
    )
    visual_datasets: str = field(
        default=None,
        metadata={"help": "visual_datasets"},
    )
    speech_datasets: str = field(
        default=None,
        metadata={"help": "speech_datasets"},
    )
    music_datasets: str = field(
        default=None,
        metadata={"help": "music_datasets"},
    )
    cache_dir: Optional[str] = field(
        default="/mnt/petrelfs/zhanjun.p/mllm/data/cache/sft",
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
        default=26,
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
    
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


from m_utils.conversation import get_conv_template

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    add_eos_token=True
) -> Dict:
    conv = get_conv_template("AnyGPT")
    roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    source_conv = sources["conversation"]
    for i, source in enumerate(source_conv):
        if roles[source[0]["role"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            try:
                assert role == conv.roles[j % 2], f"{i}"
            except:
                print(i, j, role, conv.roles[j % 2])
                print(source)
                print("*****")
                print(sources)
                print("*****")
                return
            conv.append_message(role, sentence["message"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    result = tokenizer(
        conversations,
        return_tensors=None,
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    if (
        input_ids[0][-1] != tokenizer.eos_token_id
        and len(input_ids[0]) < tokenizer.model_max_length
        and add_eos_token
    ):
        for i in range(len(input_ids)):
            # tensor加一个元素
            input_ids[i].append(tokenizer.eos_token_id)
            attention_mask[i].append(1)
    targets = copy.deepcopy(input_ids)
    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + '\n' + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = len(target)
        turns = conversation.split(conv.sep2)
        cur_len = 1
        # 对列表批量修改，
        for i in range(cur_len):
            target[i] = IGNORE_TOKEN_ID        
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            for k in range(cur_len, cur_len+instruction_len):
                target[k] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1
        excepted_len = total_len - 1 if add_eos_token else total_len
        for k in range(cur_len, excepted_len):
            target[k] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != excepted_len:
                for k in range(total_len):
                    target[k] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
    )


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    result = {}
    input_ids = examples["input_ids"]
    labels = examples["labels"]
    grouped_input_ids = []
    grouped_labels = []
    current_ids_chunk = []
    current_labels_chunk = []
    current_length = 0
    for i in range(len(input_ids)):
        if current_length + len(input_ids[i]) > data_args.block_size:
            grouped_input_ids.append(current_ids_chunk)
            grouped_labels.append(current_labels_chunk)
            current_ids_chunk = []
            current_labels_chunk = []
            current_length = 0
        current_ids_chunk.extend(input_ids[i])
        current_labels_chunk.extend(labels[i])
        current_length += len(input_ids[i])
    # Make sure the last chunk is added
    if current_length > 0:
        grouped_input_ids.append(current_ids_chunk)
        grouped_labels.append(current_labels_chunk)
    result["input_ids"] = grouped_input_ids
    result["labels"] = grouped_labels
    result["attention_mask"] = [[1 for _ in row] for row in result["input_ids"]]
    return result


def make_supervised_data_module(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset_name
    dataset_name = data_path.split("/")[-1].split(".")[0]
    # print(dataset_name)
    print(f"loading {dataset_name} tokenized data")    
    tokenized_cache_file_names = {
        "train":os.path.join(data_args.cache_dir, dataset_name, 'tokenized', 'train', 'processed_train.arrow'),
        "test":os.path.join(data_args.cache_dir, dataset_name, 'tokenized', 'test', 'processed_valid.arrow'),
    }
    raw_datasets = load_dataset("json", data_files=data_path)
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            lambda x: preprocess(x, tokenizer),
            batched=True,
            batch_size=10,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
            cache_file_names=tokenized_cache_file_names
        )
    # shuffle
    tokenized_datasets = tokenized_datasets.shuffle(seed=seed)
    tokenized_datasets = tokenized_datasets['train']
    if 'id' in tokenized_datasets.column_names:
        tokenized_datasets=tokenized_datasets.remove_columns('id')
    tokenized_datasets = tokenized_datasets.remove_columns(["conversation"])
    if training_args.val_set_size > 0:
        train_val = tokenized_datasets.train_test_split(
            test_size=training_args.val_set_size, shuffle=True, seed=seed
        )
        val_data = train_val["test"]
        tokenized_datasets = train_val["train"]
    else:
        val_data = None
    
    group_cache_file_name = os.path.join(data_args.cache_dir, dataset_name, 'group', 'train', 'processed_train.arrow')
    rank0_print(f"loading {dataset_name} group data")
    with training_args.main_process_first(desc="grouping tokens together"):
        if data_args.concatenating and dataset_name in ['journeydb', 'laion_aesthetics_6plus_8m']:
            group_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=True,
                desc=f"block size {data_args.block_size}",
                cache_file_name=group_cache_file_name
            )
        else:
            # print("not concatenating")
            group_datasets = tokenized_datasets
        train_data = group_datasets

    return train_data, val_data
        


def train():
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
    for token in special_tokens:
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
        
    if data_args.mmi_datasets != None:
        mmi_train_data_={}
        mmi_val_data={}
        datasets = data_args.mmi_datasets.split('\n')
        for dataset_path in datasets:
            if dataset_path == '':
                continue
            dataset_name = dataset_path.split("/")[-1].split(".")[0]
            train_data_, val_data_ = make_supervised_data_module(tokenizer=tokenizer, data_path=dataset_path)
            mmi_train_data_[dataset_name] = train_data_
            mmi_val_data[dataset_name] = val_data_
        mmi_train_data = concatenate_datasets(list(mmi_train_data_.values()))
    
    if data_args.speech_conv_datasets != None:
        speech_train_data_ = {}
        speech_val_data = {}
        datasets = data_args.speech_conv_datasets.split('\n')
        # print(datasets)
        for dataset_path in datasets:
            if dataset_path == '':
                continue
            dataset_name = dataset_path.split("/")[-1].split(".")[0]
            train_data_, val_data_ = make_supervised_data_module(tokenizer=tokenizer, data_path=dataset_path)
            speech_train_data_[dataset_name] = train_data_
            speech_val_data[dataset_name] = val_data_
        speech_train_data=concatenate_datasets(list(speech_train_data_.values()))
        
    if data_args.visual_datasets != None:
        visual_train_data = {}
        visual_val_data = {}
        datasets = data_args.visual_datasets.split('\n')
        # print(datasets)
        for dataset_path in datasets:
            if dataset_path == '':
                continue
            dataset_name = dataset_path.split("/")[-1].split(".")[0]
            train_data_, val_data_ = make_supervised_data_module(tokenizer=tokenizer, data_path=dataset_path)
            visual_train_data[dataset_name] = train_data_
            visual_val_data[dataset_name] = val_data_
            
        # ['instructpix2pix','laion_aesthetics_6plus_8m','journeydb','lvis_mix_680k','magicbrush','svit']
        # 拼接instructpix2pix和magicbrush: 0.3M
        t2i_datasets_name=['journeydb', 'laion_aesthetics_6plus_8m']
        t2i_datasets=[]
        other_visual_datasets=[]
        for dataset_name, dataset in visual_train_data.items():
            if dataset_name in t2i_datasets_name:
                t2i_datasets.append(dataset)
            else:
                other_visual_datasets.append(dataset)
        text2image_data = concatenate_datasets(t2i_datasets)
        other_visual_data = concatenate_datasets(other_visual_datasets)
    
    if data_args.visual_datasets != None and data_args.mmi_datasets != None and data_args.speech_conv_datasets != None and data_args.speech_datasets != None and data_args.music_datasets != None:
        probabilities=[0.28, 0.14, 0.29, 0.29]
        probabilities=[0.2, 0.1, 0.3, 0.1, 0.15, 0.15]
        train_data = interleave_datasets([other_visual_data, text2image_data, mmi_train_data, speech_train_data], probabilities=probabilities)
        val_data = {}
        for key in visual_val_data.keys():
            val_data[key] = visual_val_data[key]
        for key in speech_val_data.keys():
            val_data[key] = speech_val_data[key]
        for key in mmi_val_data.keys():
            val_data[key] = mmi_val_data[key]

    elif data_args.visual_datasets != None and data_args.mmi_datasets != None and data_args.speech_conv_datasets != None:
        probabilities=[0.28, 0.14, 0.29, 0.29]
        probabilities=[0.3, 0.15, 0.45, 0.1]
        train_data = interleave_datasets([other_visual_data, text2image_data, mmi_train_data, speech_train_data], probabilities=probabilities)
        val_data = {}
        for key in visual_val_data.keys():
            val_data[key] = visual_val_data[key]
        for key in speech_val_data.keys():
            val_data[key] = speech_val_data[key]
        for key in mmi_val_data.keys():
            val_data[key] = mmi_val_data[key]
    elif data_args.visual_datasets != None:
        probabilities=[0.7, 0.3]
        train_data = interleave_datasets([other_visual_data, text2image_data], probabilities=probabilities)
        val_data = {}
        for key in visual_val_data.keys():
            val_data[key] = visual_val_data[key]
    elif data_args.mmi_datasets != None and data_args.speech_conv_datasets != None:
        train_data = interleave_datasets([speech_train_data, mmi_train_data], probabilities=[0.25, 0.75])
        val_data = {}
        for key in speech_val_data.keys():
            val_data[key] = speech_val_data[key]
        for key in mmi_val_data.keys():
            val_data[key] = mmi_val_data[key]
    elif data_args.mmi_datasets != None:
        train_data = mmi_train_data
        val_data = mmi_val_data
    elif data_args.speech_conv_datasets != None:
        train_data = speech_train_data
        val_data = speech_val_data
    
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
    