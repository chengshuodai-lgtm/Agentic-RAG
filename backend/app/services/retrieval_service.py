from typing import List, Dict, Any, Optional, Tuple
import logging
from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder
import numpy as np

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

from ..core.config import settings
from ..core.database import get_vector_store

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.reranker = None
        self._init_reranker()
    
    def _init_reranker(self):
        """初始化重排序器"""
        try:
            # 使用sentence-transformers的CrossEncoder
            self.reranker = CrossEncoder(
                settings.RERANKER_MODEL,
                max_length=512
            )
            logger.info(f"重排序模型 {settings.RERANKER_MODEL} 加载成功")
        except Exception as e:
            logger.warning(f"重排序模型加载失败，将不使用重排序: {e}")
            self.reranker = None
    
    def hybrid_retrieval(self, 
                        query: str, 
                        collection_name: str = "default",
                        top_k: int = None,
                        use_reranker: bool = True) -> Tuple[List[Dict[str, Any]], List[str]]:
        """混合检索（向量 + 关键词）"""
        try:
            # 获取向量存储
            vector_store = get_vector_store(
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            
            if not vector_store:
                return [], ["向量数据库未初始化"]
            
            # 向量检索
            top_k = top_k or settings.RETRIEVAL_TOP_K
            vector_results = vector_store.similarity_search_with_score(
                query, k=top_k * 2
            )
            
            # 转换为字典格式
            documents = []
            for doc, score in vector_results:
                doc_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "type": "vector"
                }
                documents.append(doc_dict)
            
            # 重排序
            if use_reranker and self.reranker and documents:
                reranked_docs = self._rerank_documents(query, documents, top_k)
                return reranked_docs[:top_k], []
            
            return documents[:top_k], []
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return [], [f"检索错误: {str(e)}"]
    
    def _rerank_documents(self, 
                         query: str, 
                         documents: List[Dict[str, Any]], 
                         top_k: int) -> List[Dict[str, Any]]:
        """对文档进行重排序"""
        try:
            # 准备重排序数据
            pairs = [(query, doc["content"]) for doc in documents]
            
            # 计算重排序分数
            scores = self.reranker.predict(pairs)
            
            # 更新分数并排序
            for i, score in enumerate(scores):
                documents[i]["rerank_score"] = float(score)
                # 综合分数（结合向量分数和重排序分数）
                vector_score = documents[i].get("score", 0)
                documents[i]["combined_score"] = 0.3 * vector_score + 0.7 * float(score)
            
            # 按综合分数排序
            documents.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            
            return documents[:top_k]
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents
    
    def judge_retrieval_need(self, query: str, history: List[Dict[str, str]]) -> Tuple[bool, str]:
        """判断是否需要检索"""
        judge_prompt = f"""
        请判断以下用户查询是否需要从知识库中检索信息来回答。
        
        用户查询: {query}
        对话历史: {history[-3:] if history else "无"}
        
        需要检索的情况:
        1. 查询涉及具体事实、数据、技术细节
        2. 查询需要最新的、文档中的特定信息
        3. 查询包含专业概念、术语
        4. 查询需要引用具体文档内容
        
        不需要检索的情况:
        1. 通用问候、闲聊
        2. 简单的问题澄清
        3. 关于AI助手自身能力的问题
        4. 常识性问题
        
        请只返回"需要检索"或"不需要检索"，不需要解释。
        """
        
        # 这里简化处理，实际应该调用LLM
        # 简单规则判断
        need_keywords = ["是什么", "怎样", "如何", "为什么", "步骤", "方法", "技术", "文档", "文件", "知识"]
        if any(keyword in query for keyword in need_keywords):
            return True, "查询需要知识库信息"
        return False, "查询不需要检索"
    
    def rewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """重写查询以获得更好的检索效果"""
        rewrite_prompt = f"""
        请将以下用户查询重写为更适合文档检索的形式。
        考虑对话历史，提取查询的核心信息需求。
        
        对话历史: {history[-3:] if history else "无"}
        用户查询: {query}
        
        重写要求:
        1. 保持原意
        2. 提取关键实体和概念
        3. 消除歧义
        4. 适合向量检索
        
        重写后的查询:
        """
        
        # 这里简化处理，实际应该调用LLM
        # 简单处理：去除疑问词，保留核心内容
        query_words = query.replace("?", "").replace("？", "").strip()
        return query_words