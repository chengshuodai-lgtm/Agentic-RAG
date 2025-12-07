from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    history: List[Dict[str, str]] = []
    stream: bool = False
    
    # 检索参数
    use_agent: bool = True
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "什么是机器学习？",
                "conversation_id": "conv_123",
                "history": [],
                "stream": True,
                "use_agent": True
            }
        }

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]] = []
    agent_thoughts: List[str] = []
    response_time: float
    
class StreamResponse(BaseModel):
    type: str  # chunk, sources, thoughts, done
    data: Dict[str, Any]
    
class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None
    
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True
    
class AgentThought(BaseModel):
    step: str
    thought: str
    action: Optional[str] = None
    result: Optional[str] = None