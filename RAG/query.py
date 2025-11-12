from transformers import AutoTokenizer, AutoModelForCausalLM
from . import embedding 
import torch


#=======================加载模型并打印日志=============================
#local_model_path = ".cache/hub/models--Qwen/Qwen2-0.5B/snapshots/1.0"  # 本地模型路径
print("开始加载 Qwen/Qwen2-0.5B 分词器...")  # 新增日志
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B",
                                          trust_remote_code=True)                              
print("分词器加载完成！")  # 新增日志


print("开始加载 Qwen/Qwen2-0.5B 模型...")  # 新增日志
LLM_MODEL = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    trust_remote_code=True,
    torch_dtype=torch.float16, # 使用半精度浮点数以节省显存
    device_map="cuda:0", # 强制使用第一块GPU

)
print("模型加载完成！")  # 新增日志

#=======================定义QA生成函数=============================


def QA_Generate(query: str) ->str:
#query = "电动平衡车的安全要求是什么？"
    # Query the database for relevant passages
    chunks: list[str] = embedding.query_db(query)
    # Prepare the prompt for the model
    prompt = "Please answer the question based on the following context:\n"
    prompt += f"Query: {query}\n"
    prompt += "Context:\n"   
    for c in chunks:
        prompt += c + "\n"
        prompt += "-----\n"

    # Prepare the messages for the model
    messages = [{"role": "user", "content": prompt}]   

    # Prepare the inputs for the model
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(LLM_MODEL.device)

    # Generate the output
    outputs = LLM_MODEL.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,     # 适当增加随机性（0-1，越高越灵活）
        top_p=0.9,           #  nucleus sampling，提升多样性
        do_sample=True,      # 启用采样模式（而非贪心解码）
        repetition_penalty=1.2  # 抑制重复内容
        )
    
    # Decode the output to get the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


#print(f"Thinking content: {thinking_content}")
#print(f"Content: {content}")