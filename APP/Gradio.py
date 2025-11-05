import gradio as gr
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入hf镜像环境配置
import Gradio_env 

try:
    from RAG import query
    print("成功导入 query 模块")
except Exception as e:
    print(f"导入 query 模块失败: {e}")
    sys.exit(1)

title = "Industrial Q&A Robot (Online Version)"
description = "This is a chatbot that can answer questions based on the provided specific industrial context. (Online Version - can download models from Hugging Face if needed)"

QA_ROBOT = gr.Interface(
    fn = query.QA_Generate, 
    inputs = gr.Textbox(
        label="输入问题",
        placeholder="请输入您的问题...",
        lines=3,              # 初始显示3行
        max_lines=10,        # 最多可扩展到10行
        show_copy_button=True  # 显示复制按钮
    ),
    outputs = gr.Textbox(
        label="回答",
        lines=5,              # 初始显示5行
        max_lines=20,        # 最多可扩展到20行（输出通常更长）
        show_copy_button=True  # 显示复制按钮
    ),
    title = title, 
    description = description,
    examples = [
        ["电动平衡车的安全要求是什么？"],
        ["工业机器人在制造业中的应用有哪些？"],
        ["如何提高生产线的自动化水平？"]]
)

# Launch the interface
if __name__ == "__main__":
    print("=" * 50)
    print("启动 Industrial Q&A Robot - 在线版本")
    print("=" * 50)
    
    QA_ROBOT.launch()

