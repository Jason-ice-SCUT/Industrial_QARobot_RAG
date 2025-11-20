import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from GradioApp import Gradio_env

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model


# ============================= 1. 配置参数 ==================================
MODEL_NAME = "Qwen/Qwen2-0.5B" 
DATASET_PATH = "Finetune_datasets/ft_dataset.arrow"  # 数据集
OUTPUT_DIR = "Finetune_datasets/lora_qwen_industrial_safety"  # 微调结果保存目录
LORA_R = 4  # LoRA 秩（越大效果可能越好，但显存占用越高）
LORA_ALPHA = 8  # 缩放因子（通常是 R 的 2 倍）
LORA_DROPOUT = 0.1  # LoRA  dropout 率
BATCH_SIZE = 2  # 批次大小（根据显存调整，16GB 显存推荐 2-4）
GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积（模拟更大批次）
LEARNING_RATE = 1.5e-4  # 学习率（LoRA 微调推荐 1e-4 ~ 3e-4）
MAX_SEQ_LENGTH = 256  # 最大序列长度（根据数据长度调整）
EPOCHS = 3  # 训练轮次（小数据集 3-5 轮足够）

# ============================ 2. 加载并预处理数据集 ===========================
def load_and_preprocess_dataset(dataset_path, tokenizer):
    # 加载本地数据集
    dataset = load_from_disk(dataset_path)
    print(f"数据集结构：{dataset}")
    
    # 格式化数据为 Qwen 对话模板（instruction + input -> output）
    def format_example(instruction, input_text, output_text):
        return f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""
    
  # Tokenize 函数 (修复了 labels 逻辑)
    def tokenize_function(examples):
        instructions = examples["instruction"]
        input_texts = examples["input"]
        output_texts = examples["output"]
        
        texts = [format_example(inst, inp, out) for inst, inp, out in zip(instructions, input_texts, output_texts)]
        
        # 这里同时生成 input_ids 和 attention_mask
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建 labels，默认复制 input_ids
        input_ids = tokenized["input_ids"]
        labels = input_ids.clone()
        
        # 将 padding 部分的 label 设置为 -100 (这样计算 loss 时会被忽略)
        # 注意：tokenizer.pad_token_id 对于 Qwen 可能是 None，所以我们要用 eos_token_id 或者手动指定的 pad_token
        pad_token_id = tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        
        # 将 labels 放回 tokenized 字典中
        tokenized["labels"] = labels
       
        
        return tokenized
    
    # 移除旧列，防止数据冗余
    column_names = dataset["train"].column_names if "train" in dataset else dataset.column_names
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=1 # CPU 运行时，如果不稳定可以设为 1
    )
    
    return tokenized_dataset

# ====================== 3. 配置模型（4-bit 量化节省显存）===========================
def load_model_and_tokenizer():
    # 加载 Tokenizer（Qwen2 专用）
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right" # 右对齐补全
    )
    tokenizer.pad_token = tokenizer.eos_token  # Qwen2 默认无 pad_token，需手动设置
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",  # 强制使用 CPU
        trust_remote_code=True,
        torch_dtype=torch.float32,  
        load_in_8bit=False,  
        load_in_4bit=False,
        low_cpu_mem_usage=True  # 启用低内存模式
    )
    
    # 关闭缓存和张量并行
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

# =============================== 4. 配置 LoRA =====================================
def setup_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],  # Qwen 模型的目标层（注意力+投影层）
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"  # 因果语言模型任务
    )
    
    # 应用 LoRA 到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数比例（通常 <1%）
    return model

# ================================ 5. 训练配置 ==================================
def get_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False,
        report_to="none",
        seed=42,
        load_best_model_at_end=True,
        no_cuda=True,
        disable_tqdm=False,
        # 仅保留旧版 transformers 兼容的参数
        optim="adamw_torch",  # 使用 torch 原生 AdamW，避免 accelerate 包装
        use_cpu=True              # 显式告诉 HF Trainer 使用 CPU
    )

# =============================== 6. 启动训练 =====================================
if __name__ == "__main__":
    # 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # 加载并预处理数据集
    tokenized_dataset = load_and_preprocess_dataset(DATASET_PATH, tokenizer)
    
    # 配置 LoRA
    model = setup_lora(model)
    
    # 数据整理器（用于批量处理文本）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果LM不使用掩码语言建模
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 开始训练
    print("🚀 开始 LoRA 微调...")
    trainer.train()
    
    # 保存最终 LoRA 适配器（仅几 MB，无需保存完整模型）
    trainer.save_model(f"{OUTPUT_DIR}/final_lora_adapter")
    print(f"✅ 训练完成！LoRA 适配器已保存到 {OUTPUT_DIR}/final_lora_adapter")