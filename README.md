
# Industrial_QARobot_RAG

A Retrieval-Augmented Generation (RAG) based question-answering system designed for industrial safety scenarios. It combines document retrieval with fine-tuned language models to provide accurate answers grounded in specific industrial safety documents.


## Core Features

- **Document Chunking**: Automatically splits industrial safety documents into meaningful paragraphs (removes empty segments) for efficient processing
- **Vector Embedding & Retrieval**: Uses `BAAI/bge-base-en-v1.5` model for text embedding and ChromaDB for vector storage, enabling fast similarity search
- **Fine-Tuning Support**: Implements LoRA (Low-Rank Adaptation) for lightweight domain adaptation of the `Qwen/Qwen2-0.5B` model
- **User-Friendly Interface**: Provides a Gradio web interface for intuitive question input and answer display
- **End-to-End Workflow**: From document processing and database initialization to model fine-tuning and interactive querying

## User Interface

The system features a simple and intuitive Gradio web interface for interaction:

- **Title & Description**: Clearly labeled "Industrial Q&A Robot (Online Version)" with a brief explanation of its purpose (answers questions based on specific industrial context, supports downloading models from Hugging Face if needed)
- **Input Area**: A text box labeled "输入问题" (Enter Question) for users to type their industrial safety-related queries
- **Output Area**: A text box labeled "回答" (Answer) where the generated responses are displayed
- **Action Buttons**: 
  - "Clear": Resets the input and output fields
  - "Submit": Triggers the question processing and answer generation
  - "Flag": Allows users to flag inappropriate or incorrect responses
- **Example Queries**: Pre-listed sample questions (e.g., "电动平衡车的安全要求是什么？", "电动平衡车的机械安全有哪些？") to guide users on usage

## Project Structure

| File Path | Description |
|-----------|-------------|
| `RAG/chunk_traditional.py` | Reads industrial documents (default: `data/GB+34668-2024-data.txt`) and splits them into non-empty paragraphs using `\n\n` as delimiters |
| `RAG/embedding.py` | Manages text embedding generation (via `BAAI/bge-base-en-v1.5`), creates/maintains ChromaDB vector database (`chroma.db`), and handles retrieval queries |
| `RAG/query.py` | Core QA logic: retrieves relevant document chunks, constructs prompts, and generates answers using `Qwen/Qwen2-0.5B` with LoRA adapters |
| `RAG/data_processor.py` | Processes raw QA pairs (from `Finetune_datasets/ft_QA_data.txt`) into formatted datasets for fine-tuning, supports saving to local or pushing to Hugging Face Hub |
| `RAG/LoRA_FT.py` | Configures and runs LoRA fine-tuning for `Qwen/Qwen2-0.5B`, with adjustable parameters (batch size, learning rate, epochs, etc.) |
| `GradioApp/Gradio.py` | Web interface for user interaction, automatically resets and rebuilds the vector database on launch |


## Quick Start

### 1. Prepare Data
- Place industrial safety documents in the `data` directory (default file: `GB+34668-2024-data.txt`)
- For model fine-tuning, add QA pairs to `Finetune_datasets/ft_QA_data.txt` in the format: `问题：[your_question] 答案：[your_answer]`


### 2. Initialize Vector Database
Build the vector database from your documents:
```bash
python RAG/embedding.py
```
This creates a `chroma.db` directory storing embedded document chunks for fast retrieval.


### 3. Fine-Tune Model (Optional)
To adapt the model to your industrial domain:
```bash
# Process raw QA data into fine-tuning format
python RAG/data_processor.py

# Run LoRA fine-tuning (saves adapter to `Finetune_datasets/lora_qwen_industrial_safety`)
python RAG/LoRA_FT.py
```


### 4. Launch Interactive Interface
Start the web-based QA interface:
```bash
python GradioApp/Gradio.py
```
Access the interface via the local URL displayed (default: `http://localhost:7860`).


## Example Queries

- 电动平衡车的安全要求是什么？ (What are the safety requirements for electric balance bikes?)
- 电动平衡车的机械安全有哪些？ (What mechanical safety aspects apply to electric balance bikes?)
- 什么是翘板功能？ (What is the rocker function?)


## Notes

- **First Run**: Initial model downloads (`Qwen/Qwen2-0.5B`, `BAAI/bge-base-en-v1.5`) may take time depending on network speed
- **Hardware**: CPU is supported but GPU acceleration is recommended for faster fine-tuning and inference
- **Database**: Automatically resets when launching Gradio to ensure latest document changes are reflected
- **Fine-Tuning**: Adjust parameters like `BATCH_SIZE` and `MAX_SEQ_LENGTH` in `LoRA_FT.py` based on available memory
```
