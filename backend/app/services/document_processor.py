from typing import List, Optional, Dict, Any
import os
from pathlib import Path
import uuid
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.schema import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from ..core.config import settings
from ..core.database import get_vector_store

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[LangchainDocument]:
        """加载PDF文档"""
        try:
            # 使用UnstructuredPDFLoader以获得更好的解析效果
            loader = UnstructuredPDFLoader(file_path, mode="elements")
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 共{len(documents)}个元素")
            return documents
        except Exception as e:
            logger.error(f"PDF加载失败: {e}")
            # 回退到PyPDFLoader
            loader = PyPDFLoader(file_path)
            return loader.load()
    
    def split_documents(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """分割文档为块"""
        return self.text_splitter.split_documents(documents)
    
    def add_metadata(self, documents: List[LangchainDocument], 
                    source: str, file_type: str) -> List[LangchainDocument]:
        """为文档块添加元数据"""
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "chunk_id": str(uuid.uuid4()),
                "source": source,
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": len(documents)
            })
        return documents
    
    def process_and_store(self, file_path: str, collection_name: str = "default") -> Dict[str, Any]:
        """处理并存储文档到向量数据库"""
        try:
            # 加载文档
            raw_docs = self.load_pdf(file_path)
            
            # 分割文档
            split_docs = self.split_documents(raw_docs)
            
            # 添加元数据
            enhanced_docs = self.add_metadata(
                split_docs, 
                source=os.path.basename(file_path),
                file_type="pdf"
            )
            
            # 获取向量存储
            vector_store = get_vector_store(
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            
            # 添加文档
            ids = vector_store.add_documents(enhanced_docs)
            
            # 获取统计信息
            collection = vector_store._collection
            count = collection.count() if collection else 0
            
            return {
                "status": "success",
                "total_chunks": len(enhanced_docs),
                "collection": collection_name,
                "total_documents": count,
                "source": file_path
            }
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_collection_info(self, collection_name: str = "default") -> Dict[str, Any]:
        """获取集合信息"""
        try:
            vector_store = get_vector_store(
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            collection = vector_store._collection
            if collection:
                count = collection.count()
                metadata = collection.metadata or {}
                return {
                    "collection": collection_name,
                    "total_documents": count,
                    "metadata": metadata
                }
            return {"collection": collection_name, "total_documents": 0}
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"error": str(e)}