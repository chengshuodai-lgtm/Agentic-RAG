from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
import uuid
from typing import List
import logging

from .models import (
    ChatRequest, 
    ChatResponse, 
    StreamResponse,
    SearchRequest,
    Document
)
from ..services.agent_service import AgentService
from ..services.document_processor import DocumentProcessor
from ..services.retrieval_service import RetrievalService
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# 初始化服务
agent_service = AgentService()
doc_processor = DocumentProcessor()
retriever = RetrievalService()

# 内存存储对话历史（生产环境应该用数据库）
conversation_store = {}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天端点（非流式）"""
    try:
        start_time = time.time()
        
        # 生成或获取会话ID
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        
        # 获取Agent响应
        response_data = None
        for data in agent_service.process_query(
            query=request.message,
            conversation_id=conversation_id,
            history=request.history,
            use_agent=request.use_agent,
            stream=False
        ):
            if data["type"] == "complete":
                response_data = data["data"]
            elif data["type"] == "error":
                raise HTTPException(status_code=500, detail=data["data"]["error"])
        
        if not response_data:
            raise HTTPException(status_code=500, detail="未生成响应")
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 更新对话历史
        if conversation_id not in conversation_store:
            conversation_store[conversation_id] = []
        
        conversation_store[conversation_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        conversation_store[conversation_id].append({
            "role": "assistant",
            "content": response_data["response"],
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=response_data["response"],
            conversation_id=conversation_id,
            sources=response_data.get("sources", []),
            agent_thoughts=[t.dict() for t in response_data.get("thoughts", [])],
            response_time=response_time
        )
        
    except Exception as e:
        logger.error(f"聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天端点"""
    async def event_generator():
        try:
            conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
            
            full_response = ""
            sources = []
            thoughts = []
            
            for data in agent_service.process_query(
                query=request.message,
                conversation_id=conversation_id,
                history=request.history,
                use_agent=request.use_agent,
                stream=True
            ):
                if data["type"] == "chunk":
                    chunk = data["data"]["text"]
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'data': {'text': chunk}})}\n\n"
                
                elif data["type"] == "thought":
                    thought = data["data"]
                    thoughts.append(thought)
                    yield f"data: {json.dumps({'type': 'thought', 'data': thought})}\n\n"
                
                elif data["type"] == "complete":
                    sources = data["data"].get("sources", [])
                    thoughts = data["data"].get("thoughts", [])
                    
                    # 发送完成信号
                    yield f"data: {json.dumps({'type': 'complete', 'data': {'response': full_response}})}\n\n"
                    
                    # 发送源文档
                    for i, source in enumerate(sources[:3]):
                        source_data = {
                            "index": i + 1,
                            "content": source.get("content", "")[:200] + "...",
                            "metadata": source.get("metadata", {}),
                            "score": source.get("score", 0)
                        }
                        yield f"data: {json.dumps({'type': 'source', 'data': source_data})}\n\n"
                    
                    # 发送思考过程总结
                    yield f"data: {json.dumps({'type': 'thoughts_summary', 'data': {'count': len(thoughts)}})}\n\n"
                    
                    # 更新对话历史
                    if conversation_id not in conversation_store:
                        conversation_store[conversation_id] = []
                    
                    conversation_store[conversation_id].append({
                        "role": "user",
                        "content": request.message,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    conversation_store[conversation_id].append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    break
                
                elif data["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'data': {'message': data['data']['error']}})}\n\n"
                    break
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"流式响应错误: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/documents/upload")
async def upload_document(file_path: str, collection_name: str = "default"):
    """上传并处理文档"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        result = doc_processor.process_and_store(file_path, collection_name)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "message": f"文档处理完成，添加{result['total_chunks']}个块",
            "collection": result["collection"],
            "total_documents": result["total_documents"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    """获取集合信息"""
    try:
        info = doc_processor.get_collection_info(collection_name)
        
        if "error" in info:
            raise HTTPException(status_code=500, detail=info["error"])
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_documents(request: SearchRequest):
    """搜索文档"""
    try:
        documents, errors = retriever.hybrid_retrieval(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker
        )
        
        if errors:
            logger.warning(f"检索错误: {errors}")
        
        return {
            "query": request.query,
            "results": documents,
            "total": len(documents),
            "errors": errors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """获取对话历史"""
    if conversation_id not in conversation_store:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversation_store[conversation_id]
    }

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除对话历史"""
    if conversation_id in conversation_store:
        del conversation_store[conversation_id]
    
    return {"status": "success", "message": "对话已删除"}