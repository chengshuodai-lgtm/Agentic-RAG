import requests
import json
from typing import Dict, List, Optional, Any
import streamlit as st

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    def _handle_response(self, response):
        """处理API响应"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP错误: {e}"
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", error_msg)
            except:
                pass
            raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"请求失败: {e}")
    
    def chat(self, 
            message: str,
            conversation_id: Optional[str] = None,
            history: List[Dict] = None,
            use_agent: bool = True,
            **kwargs) -> Dict:
        """发送聊天消息（非流式）"""
        url = f"{self.api_base}/chat"
        
        payload = {
            "message": message,
            "conversation_id": conversation_id,
            "history": history or [],
            "use_agent": use_agent,
            "stream": False
        }
        
        # 添加其他参数
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            return self._handle_response(response)
        except Exception as e:
            raise Exception(f"聊天请求失败: {e}")
    
    def chat_stream(self, 
                   message: str,
                   conversation_id: Optional[str] = None,
                   history: List[Dict] = None,
                   use_agent: bool = True,
                   **kwargs) -> requests.Response:
        """发送聊天消息（流式）"""
        url = f"{self.api_base}/chat/stream"
        
        payload = {
            "message": message,
            "conversation_id": conversation_id,
            "history": history or [],
            "use_agent": use_agent,
            "stream": True
        }
        
        # 添加其他参数
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            return response
        except Exception as e:
            raise Exception(f"流式聊天请求失败: {e}")
    
    def upload_document(self, file_path: str, collection_name: str = "default") -> Dict:
        """上传文档"""
        url = f"{self.api_base}/documents/upload"
        
        params = {
            "file_path": file_path,
            "collection_name": collection_name
        }
        
        try:
            response = requests.post(url, params=params, timeout=60)
            return self._handle_response(response)
        except Exception as e:
            raise Exception(f"文档上传失败: {e}")
    
    def get_collection_info(self, collection_name: str = "default") -> Dict:
        """获取集合信息"""
        url = f"{self.api_base}/documents/collections/{collection_name}"
        
        try:
            response = requests.get(url, timeout=10)
            return self._handle_response(response)
        except Exception as e:
            raise Exception(f"获取集合信息失败: {e}")
    
    def search(self, query: str, top_k: int = 5, use_reranker: bool = True) -> Dict:
        """搜索文档"""
        url = f"{self.api_base}/search"
        
        payload = {
            "query": query,
            "top_k": top_k,
            "use_reranker": use_reranker
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            return self._handle_response(response)
        except Exception as e:
            raise Exception(f"搜索失败: {e}")
    
    def get_conversation(self, conversation_id: str) -> Dict:
        """获取对话历史"""
        url = f"{self.api_base}/conversations/{conversation_id}"
        
        try:
            response = requests.get(url, timeout=10)
            return self._handle_response(response)
        except Exception as e:
            raise Exception(f"获取对话失败: {e}")
    
    def delete_conversation(self, conversation_id: str) -> Dict:
        """删除对话历史"""
        url = f"{self.api_base}/conversations/{conversation_id}"
        
        try:
            response = requests.delete(url, timeout=10)
            return self._handle_response(response)
        except Exception as e:
            raise Exception(f"删除对话失败: {e}")
    
    def health_check(self) -> Dict:
        """健康检查"""
        url = f"{self.base_url}/health"
        
        try:
            response = requests.get(url, timeout=5)
            return self._handle_response(response)
        except Exception as e:
            raise Exception(f"健康检查失败: {e}")