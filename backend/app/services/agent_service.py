from typing import List, Dict, Any, Optional, Generator, Tuple
import logging
from datetime import datetime
import json

from .llm_service import LLMService
from .retrieval_service import RetrievalService
from ..api.models import AgentThought

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self):
        self.llm = LLMService()
        self.retriever = RetrievalService()
        self.conversation_memory = {}
    
    def process_query(self, 
                     query: str,
                     conversation_id: str,
                     history: List[Dict[str, str]] = None,
                     use_agent: bool = True,
                     stream: bool = False) -> Generator[Dict[str, Any], None, None]:
        """处理用户查询（Agentic RAG核心逻辑）"""
        
        thoughts = []
        all_documents = []
        
        try:
            # 1. 记录开始
            thoughts.append(AgentThought(
                step="start",
                thought=f"开始处理查询: {query}"
            ))
            
            if stream:
                yield {"type": "thought", "data": thoughts[-1].dict()}
            
            if not use_agent:
                # 简单RAG模式
                response, docs, agent_thoughts = self._simple_rag(query, history or [])
                thoughts.extend(agent_thoughts)
                all_documents.extend(docs)
                
                if stream:
                    for chunk in self.llm.generate(response, stream=True):
                        yield {"type": "chunk", "data": {"text": chunk}}
                else:
                    yield {"type": "complete", "data": {"response": response, "thoughts": thoughts, "sources": docs}}
                return
            
            # 2. 查询重写
            rewritten_query = self.retriever.rewrite_query(query, history or [])
            thoughts.append(AgentThought(
                step="query_rewrite",
                thought=f"原始查询: {query}",
                action="重写查询",
                result=f"重写后: {rewritten_query}"
            ))
            
            if stream:
                yield {"type": "thought", "data": thoughts[-1].dict()}
            
            # 3. 检索必要性判断
            need_retrieval, reason = self.retriever.judge_retrieval_need(rewritten_query, history or [])
            thoughts.append(AgentThought(
                step="retrieval_judgment",
                thought="判断是否需要检索",
                action="分析查询类型",
                result=f"{'需要检索' if need_retrieval else '不需要检索'}: {reason}"
            ))
            
            if stream:
                yield {"type": "thought", "data": thoughts[-1].dict()}
            
            context = ""
            retrieved_docs = []
            
            if need_retrieval:
                # 4. 多轮检索
                for turn in range(3):  # 最多3轮检索
                    thoughts.append(AgentThought(
                        step=f"retrieval_turn_{turn+1}",
                        thought=f"第{turn+1}轮检索",
                        action="执行检索",
                        result=f"检索查询: {rewritten_query}"
                    ))
                    
                    if stream:
                        yield {"type": "thought", "data": thoughts[-1].dict()}
                    
                    # 执行检索
                    docs, errors = self.retriever.hybrid_retrieval(
                        rewritten_query,
                        top_k=5
                    )
                    
                    if errors:
                        thoughts.append(AgentThought(
                            step="retrieval_error",
                            thought="检索遇到错误",
                            action="处理错误",
                            result=str(errors)
                        ))
                    
                    retrieved_docs.extend(docs)
                    all_documents.extend(docs)
                    
                    # 构建上下文
                    context_parts = []
                    for i, doc in enumerate(docs[:3]):  # 取前三相关文档
                        content = doc["content"][:500]  # 截断
                        source = doc["metadata"].get("source", "未知")
                        context_parts.append(f"[文档{i+1} - {source}]:\n{content}")
                    
                    if context_parts:
                        context = "\n\n".join(context_parts)
                        thoughts.append(AgentThought(
                            step="context_building",
                            thought="构建检索上下文",
                            action="整合检索结果",
                            result=f"收集到{len(docs)}个相关文档片段"
                        ))
                        
                        if stream:
                            yield {"type": "thought", "data": thoughts[-1].dict()}
                    
                    # 判断是否需要继续检索
                    if self._should_continue_retrieval(query, context, turn):
                        # 生成深化查询
                        deeper_query = self._generate_deeper_query(query, context)
                        rewritten_query = deeper_query
                        thoughts.append(AgentThought(
                            step="query_deepening",
                            thought="深化查询以获得更相关信息",
                            action="生成新查询",
                            result=f"新查询: {deeper_query}"
                        ))
                        
                        if stream:
                            yield {"type": "thought", "data": thoughts[-1].dict()}
                    else:
                        break
            
            # 5. 生成回答
            thoughts.append(AgentThought(
                step="generation",
                thought="开始生成回答",
                action="调用LLM生成最终回答"
            ))
            
            if stream:
                yield {"type": "thought", "data": thoughts[-1].dict()}
            
            # 构建系统提示
            system_prompt = self._build_system_prompt(context, retrieved_docs)
            
            # 构建用户提示
            user_prompt = self._build_user_prompt(query, context, history or [])
            
            # 生成回答
            if stream:
                full_response = ""
                for chunk in self.llm.generate(
                    user_prompt,
                    system_prompt=system_prompt,
                    stream=True
                ):
                    if chunk:
                        full_response += chunk
                        yield {"type": "chunk", "data": {"text": chunk}}
                
                thoughts.append(AgentThought(
                    step="complete",
                    thought="回答生成完成",
                    action="整理结果",
                    result=f"生成{len(full_response)}字符的回答"
                ))
                
                yield {
                    "type": "complete",
                    "data": {
                        "response": full_response,
                        "thoughts": thoughts,
                        "sources": all_documents[:5]  # 返回前5个相关文档
                    }
                }
            else:
                response = self.llm.get_completion(
                    user_prompt,
                    system_prompt=system_prompt
                )
                
                thoughts.append(AgentThought(
                    step="complete",
                    thought="回答生成完成",
                    action="整理结果",
                    result=f"生成{len(response)}字符的回答"
                ))
                
                yield {
                    "type": "complete",
                    "data": {
                        "response": response,
                        "thoughts": thoughts,
                        "sources": all_documents[:5]
                    }
                }
            
        except Exception as e:
            logger.error(f"Agent处理失败: {e}")
            error_thought = AgentThought(
                step="error",
                thought="处理过程发生错误",
                action="错误处理",
                result=str(e)
            )
            thoughts.append(error_thought)
            
            yield {
                "type": "error",
                "data": {
                    "error": str(e),
                    "thoughts": thoughts
                }
            }
    
    def _simple_rag(self, query: str, history: List[Dict[str, str]]) -> Tuple[str, List, List]:
        """简单RAG处理"""
        thoughts = []
        
        # 检索
        docs, errors = self.retriever.hybrid_retrieval(query)
        
        thoughts.append(AgentThought(
            step="simple_retrieval",
            thought="执行检索",
            result=f"检索到{len(docs)}个相关文档"
        ))
        
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(docs[:3]):
            content = doc["content"][:500]
            source = doc["metadata"].get("source", "未知")
            context_parts.append(f"[文档{i+1} - {source}]:\n{content}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # 生成回答
        system_prompt = f"""你是一个智能助手，基于提供的上下文信息回答问题。
        
        可用上下文：
        {context}
        
        如果上下文不足以回答问题，请说明你不知道。不要编造信息。"""
        
        user_prompt = f"问题：{query}"
        
        if history:
            conversation = "\n".join([f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-3:]])
            user_prompt = f"对话历史：\n{conversation}\n\n当前问题：{query}"
        
        response = self.llm.get_completion(user_prompt, system_prompt=system_prompt)
        
        return response, docs, thoughts
    
    def _should_continue_retrieval(self, query: str, context: str, turn: int) -> bool:
        """判断是否需要继续检索"""
        if turn >= 2:  # 最多3轮
            return False
        
        if not context:
            return True
        
        # 简单判断：如果上下文长度小于阈值，继续检索
        if len(context) < 1000:
            return True
        
        return False
    
    def _generate_deeper_query(self, original_query: str, context: str) -> str:
        """生成深化查询"""
        prompt = f"""
        基于原始查询和已获取的上下文，生成一个更深入、更具体的查询来获取更多相关信息。
        
        原始查询: {original_query}
        已获取上下文: {context[:1000]}...
        
        请生成一个新的查询：
        """
        
        new_query = self.llm.get_completion(prompt, max_tokens=100)
        return new_query.strip() or original_query
    
    def _build_system_prompt(self, context: str, docs: List[Dict]) -> str:
        """构建系统提示"""
        if not context:
            return """你是一个有帮助的AI助手。请以专业、准确的方式回答问题。"""
        
        sources_info = []
        for i, doc in enumerate(docs[:3]):
            source = doc["metadata"].get("source", "未知文档")
            page = doc["metadata"].get("page", "未知页码")
            sources_info.append(f"来源{i+1}: {source} (页面: {page})")
        
        sources_str = "\n".join(sources_info)
        
        return f"""你是一个基于文档的智能助手。请严格基于提供的上下文信息回答问题。

可用上下文：
{context}

相关信息来源：
{sources_str}

回答要求：
1. 严格基于上下文信息，不要编造
2. 如果上下文没有相关信息，请说明你不知道
3. 引用具体来源（如：根据来源1的信息...）
4. 回答要准确、专业、有用
5. 如果问题与上下文无关，请基于你的知识回答，但要说明这不是来自文档"""
    
    def _build_user_prompt(self, query: str, context: str, history: List[Dict]) -> str:
        """构建用户提示"""
        if not history:
            return f"问题：{query}"
        
        # 包含最近3轮历史
        history_parts = []
        for h in history[-3:]:
            role = h.get("role", "user")
            content = h.get("content", "")
            history_parts.append(f"{role}: {content}")
        
        history_str = "\n".join(history_parts)
        
        if context:
            return f"""对话历史：
{history_str}

当前问题：{query}

请基于上下文信息回答问题。"""
        else:
            return f"""对话历史：
{history_str}

当前问题：{query}

请回答问题。"""