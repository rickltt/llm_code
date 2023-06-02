import os 
import torch
import logging
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tokenizer, model = None, None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/disc1/models/chatglm-6b")
    cache_dir: Optional[str] = field(default="cache_data")

@dataclass
class DataTrainingArguments:
    max_seq_length: Optional[int] = field(default=512)
    train_file: Optional[str] = field(default="/home/lingtongtao/code/ChatGLM-6B/ptuning/AdvertiseGen/train.json")
    validation_file: Optional[str] = field(default="/home/lingtongtao/code/ChatGLM-6B/ptuning/AdvertiseGen/dev.json")
    test_file: Optional[str] = field(default=None)
    prompt_column: Optional[str] = field(default="content")
    response_column: Optional[str] = field(default="content")
    overwrite_cache: bool = field(default=True)

class MyTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        def save_tunable_parameters(model, path):
            saved_params = {
                k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
            }
            # saved_params = model.state_dict()
            torch.save(saved_params, path)

        save_tunable_parameters(
            self.model, os.path.join(output_dir, "chatglm-lora.pt")
        )

def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1)
            + ids[(seq_len - 1) :]
            + [tokenizer.pad_token_id]
            + [-100] * (longest - ids_l - 1)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        attention_mask, position_ids = get_masks_and_position_ids(
            ids, seq_len, longest, _ids.device, gmask=False
        )
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def format_example(example: dict) -> dict:
    return example


def main():
    global tokenizer 
    global model
    model_args, data_args, training_args = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)).parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_auth_token = True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_auth_token = True).half().cuda()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
        target_modules=['query_key_value',],
    )
    model = get_peft_model(model, peft_config)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    dataset = load_dataset(
        extension,
        data_files=data_files,
        cache_dir= model_args.cache_dir
    )
    dataset["train"] = dataset["train"].select(range(100))
    dataset["validation"] = dataset["validation"].select(range(10))
    print(dataset)
    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column

    def preprocess(example):
        prompt = example[prompt_column]
        target = example[response_column]
        prompt_ids = tokenizer.encode(prompt, max_length=data_args.max_seq_length, truncation=True)
        target_ids = tokenizer.encode(target, max_length=data_args.max_seq_length, truncation=True, add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        return {"input_ids": input_ids, "seq_len": len(prompt_ids)}
    
    def filter_nan(example):
        return example[prompt_column] is not None and example[response_column] is not  None
    
    tokenized_datasets = dataset.map(function=format_example).filter(function=filter_nan)
    tokenized_datasets = tokenized_datasets.map(function=preprocess, load_from_cache_file=not data_args.overwrite_cache,
                                                desc="Running tokenizer on dataset")

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    trainer.train()

if __name__ == '__main__':
    main()