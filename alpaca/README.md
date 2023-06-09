# Chinese-LLaMA-Alpaca

## 下载LLaMA模型文件(7B,13B,30B,65B)

```shell
# 如果不想下载30B或者65B，注释掉N_SHARD_DICT["30B"]="3"或者N_SHARD_DICT["65B"]="7"
bash download_llama.sh
```
## 将原始LLaMA模型转成HuggingFace格式

将原版LLaMA的`tokenizer.model`放在`--input_dir`指定的目录，其余文件放在`${input_dir}/${model_size}`下，`--model_size`可选`7B,13B,30B,65B`，`--output_dir`中将存放转换好的HF版权重。

```python
python utils/convert_llama_weights_to_hf.py \
    --input_dir path_to_original_llama_root_dir \
    --model_size 7B \
    --output_dir path_to_original_llama_hf_dir
```

## 词表扩充

原版LLaMA词表32K，对中文支持有限，因此需要进一步扩充词表。

```python 
python utils/merge_tokenizers.py \
  --llama_tokenizer_dir /workspace/model/llama-7b-hf-tokenizer \
  --chinese_sp_model_file /workspace/code/Chinese-LLaMA-Alpaca/scripts/chinese_sp.model
```

- `llama_tokenizer_dir`: 指向存放原版LLaMA tokenizer的目录
- `chinese_sp_model_file`: 指向用sentencepiece训练的中文词表文件，有20K中文词表，和原版LLaMA的32K词表合并，去除重复token后，词表长度为49953。

运行完后，`utils/merged_tokenizer_sp`目录下为训练好的词表模型，`utils/merged_tokenizer_hf`目录下为HF格式训练好的词表模型。

## 指令精调

指令精调数据格式采用原版的Stanford Alpaca不带input的模板，对于带input的数据，采用`f"{instruction}+\n+{input}"`的形式拼接。

```json
[
  {"instruction" : ...,
   "input" : ...,
   "output" : ...
  },
  ...
]
```

- `--tokenizer_name_or_path`: Chinese-Alpaca tokenizer所在的目录，合并词表后，在路径`utils/merged_tokenizer_hf`下
- `--dataset_dir`：指令精调数据的目录，包含一个或多个以`json`结尾的Stanford Alpaca格式的指令精调数据文件
- `--validation_file`；用作验证集的单个指令精调文件，以`json`结尾，同样遵循Stanford Alpaca格式

> learning_rate、batch_size、training_steps、eval_steps， save_steps等训练相关超参数可自己调整。

开始训练：
```
bash run_lora.sh
```
- `--model_name_or_path`: 原版HF格式LLaMA模型
- `--peft_path`: Chinese-Alpaca的LoRA权重目录
- `--lora_rank`、`--lora_alpha`、`--lora_dropout`、`--lora_trainable`和`--modules_to_save`等Lora参数可以更改


## LoRA权重与基础模型合并

接下来，将进过预训练和精调的LoRA权重合并回基础模型（注意：多个Lora权重和多个tokenizer之间使用逗号分割，两个LoRA模型的顺序很重要，不能颠倒。先写预训练Lora权重，然后写精调Lora权重），具体执行运行过程如下：

```python
python merge_llama_with_chinese_lora.py \
     --base_model /workspace/model/llama-7b-hf \
     --lora_model /workspace/output/book/lora,/workspace/output/llama-book-alpace-zh/lora \
     --output_type huggingface \
     --output_dir /workspace/output/book-alpaca-merge-hf
```

> 除此之外，也可以直接使用原始权重与模型预训练Lora权重合并之后的权重与模型精调的Lora权重进行合并。

## 模型推理
```python
python utils/inference_hf.py \
     --base_model /workspace/output/book-alpaca-merge-hf \
     --with_prompt \
     --interactive
```