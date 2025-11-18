"""分享上传到huggingface"""
import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from GradioApp import Gradio_env

from huggingface_hub import login, HfFolder
from HF_token import HF_TOKEN
from datasets import load_from_disk, Dataset
import os
import pandas as pd
import datasets

def read_ft_data() -> list[str]:

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, "Finetune_datasets", "ft_QA_data.txt")
    ft_data: str = ""
    with open(data_file_path, 'r', encoding='utf-8') as f:
        ft_data = f.read()
        ft_data_list = ft_data.splitlines()
        print(f"数据集共有 {len(ft_data_list)} 条 QA 对")
        print(ft_data_list[:5])
    return ft_data_list

def generate_qa_batch(ft_data_list: list[str]) -> dict:
    """生成QA对"""
    questions = []
    answers = []
    # 遍历列表，提取问题和答案
    for sentence in ft_data_list:  # idx从1开始计数
        try:
            # 先检查是否包含必要的分隔符
            if "问题：" not in sentence or "答案：" not in sentence:
                raise ValueError("缺少'问题：'或'答案：'标记")
            q_part, a_part = sentence.split("答案：", 1)  # 从"答案："处分割成两部分
            q = q_part.split("问题：", 1)[1].strip()  # 从问题部分提取内容
            a = a_part.strip()  # 提取答案内容
            questions.append(q)
            answers.append(a)
            print(f"=============================================")
            print(f"问题：{q}")
            print(f"答案：{a}\n")
        except:
            # 处理格式异常
            questions.append("")
            answers.append("")
            print(f"处理失败: {sentence}")
        
    return {"question": questions, "answer": answers}

def format_for_lora(dataset: datasets.Dataset):
    """转换为模型微调所需的对话格式（符合Qwen等模型的chat template）"""
    return {
        "instruction": "根据问题生成准确回答",  # 固定指令（可根据需求修改）
        "input": dataset["question"],           # 输入为生成的问题
        "output": dataset["answer"]             # 输出为对应的答案
    }

# 直接使用token登录（最可靠）
def login_with_token(token):
    """使用token直接登录"""
    try:
        # 保存token到Hugging Face配置
        HfFolder.save_token(token)
        print("✅ Token已保存到本地配置")
            
        # 验证登录
        login(token=token, add_to_git_credential=True)
        print("✅ 登录成功！")
        return True
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return False

if __name__ == '__main__':
    # 读取数据集
    QA_str: list[str] = read_ft_data()
    QA_list = generate_qa_batch(QA_str)
    QA_dataset = Dataset.from_dict(QA_list)
    print(f"数据集大小: {len(QA_dataset)}")

    lora_dataset = QA_dataset.map(format_for_lora, 
                            remove_columns=QA_dataset.column_names)  
    
    #可选：拆分训练集和验证集（按 9:1 比例）
    lora_dataset = lora_dataset.train_test_split(test_size=0.1, seed=42)

    # 保存到本地
    local_save_path = "Finetune_datasets/ft_dataset.arrow"
    lora_dataset.save_to_disk(local_save_path)
    print(f"数据集已保存到本地: {local_save_path}")

    # 可选：保存为其他格式（如JSON）
    lora_dataset['train'].to_json("Finetune_datasets/ft_train.json")
    lora_dataset['test'].to_json("Finetune_datasets/ft_test.json")
    print("数据集已保存为JSON格式")
#'''
#'''
    if login_with_token(HF_TOKEN):
        # 推送数据集
        lora_dataset.push_to_hub("Jason-ice-SCUT/Industrial_Safety_Finetune_Datasets")  # 替换为您的用户名
        print("✅ 数据集已推送到Hugging Face Hub！")
    else:
        print("请先获取正确的Hugging Face token")
#'''