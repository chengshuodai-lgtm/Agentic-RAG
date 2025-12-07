from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # 项目配置
    PROJECT_NAME: str = "Agentic RAG System"
    VERSION: str = "1.0.0"
    
    # 路径配置
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_DIR: Path = BASE_DIR / "chroma_db"
    
    # 模型配置
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    LLM_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # LLM配置
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000
    LLM_TOP_P: float = 0.9
    
    # 检索配置
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # 分块配置
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["http://localhost:8501", "http://localhost:3000"]
    
    # Agent配置
    MAX_RETRIEVAL_TURNS: int = 3
    ENABLE_QUERY_REWRITE: bool = True
    ENABLE_RETRIEVAL_JUDGE: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()

# 确保目录存在
for dir_path in [settings.DATA_DIR, settings.CHROMA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)