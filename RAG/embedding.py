from . import chunk_traditional as chunk_t
import chromadb
import os
from FlagEmbedding import FlagModel

EMBEDDING_MODEL = FlagModel("BAAI/bge-base-en-v1.5",
                            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:")
                            #use_fp16=True)

# 获取项目根目录（RAG 目录的上一级）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chroma_db_path = os.path.join(project_root, "chroma.db")

chromadb_client = chromadb.PersistentClient(chroma_db_path)
chromadb_collection = chromadb_client.get_or_create_collection("IndustrialQA_DB")

def create_db() ->None:
    for idx, c in enumerate(chunk_t.chunk_text()):
        #print(f"Process: {c}")
        embedding: list[float] = embed_text(c, for_query=True)
        chromadb_collection.upsert(
            ids=str(idx),
            documents=c,
            embeddings=embedding
        )

# Define the function to generate embeddings for a given text 
def embed_text(text: str, for_query: bool) -> list[float]:
    model = EMBEDDING_MODEL
    if for_query:
        embeddings = model.encode_queries([text])
        embeddings_list = embeddings.tolist()

        assert embeddings_list
        assert embeddings_list[0]
        return embeddings_list[0]
        
    else:
        embeddings = model.encode_corpus([text])
        embeddings_list = embeddings.tolist()

        assert embeddings_list
        assert embeddings_list[0]
        return embeddings_list[0]
        
  
def query_db(query: str) -> list[str]:
    embedding = embed_text(query, for_query=True)
    results = chromadb_collection.query(
        query_embeddings=[embedding],  # ChromaDB 需要列表格式
        n_results=3
    )
    assert results['documents']
    assert results['documents'][0]  # 确保有结果
    return results['documents'][0]  # 返回第一个查询的结果列表

if __name__ == "__main__":
    create_db()
