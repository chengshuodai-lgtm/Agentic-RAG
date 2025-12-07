from typing import Optional
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

from .config import settings

class ChromaDBManager:
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaDBManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(settings.CHROMA_DIR),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
    
    def get_client(self):
        return self._client
    
    def get_or_create_collection(self, name: str = "default"):
        """获取或创建集合"""
        try:
            # 尝试获取现有集合
            collection = self._client.get_collection(name=name)
            print(f"获取现有集合: {name}")
        except Exception as e:
            # 集合不存在，创建新集合
            print(f"创建新集合: {name}, 错误: {e}")
            collection = self._client.create_collection(
                name=name,
                metadata={"description": "Agentic RAG documents"}
            )
        return collection

def get_vector_store(collection_name: str = "default", 
                    embeddings: Optional[Embeddings] = None) -> Chroma:
    """获取向量存储实例"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    # 初始化ChromaDB管理器
    db_manager = ChromaDBManager()
    
    # 创建LangChain的Chroma实例
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.CHROMA_DIR),
        client_settings=Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(settings.CHROMA_DIR),
            anonymized_telemetry=False
        )
    )
    
    return vector_store