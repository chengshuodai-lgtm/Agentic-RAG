import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any
import asyncio
import websockets
import uuid

from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from components.config_panel import render_config_panel
from utils.api_client import APIClient

# 页面配置
st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f4fd;
    }
    .thinking-container {
        background-color: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .source-doc {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class ChatApp:
    def __init__(self):
        self.api_client = APIClient()
        self._init_session_state()
    
    def _init_session_state(self):
        """初始化会话状态"""
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "agent_thoughts" not in st.session_state:
            st.session_state.agent_thoughts = []
        
        if "sources" not in st.session_state:
            st.session_state.sources = []
        
        if "use_agent" not in st.session_state:
            st.session_state.use_agent = True
        
        if "streaming" not in st.session_state:
            st.session_state.streaming = True
        
        if "is_loading" not in st.session_state:
            st.session_state.is_loading = False
        
        if "retrieval_config" not in st.session_state:
            st.session_state.retrieval_config = {
                "top_k": 5,
                "use_reranker": True,
                "temperature": 0.1
            }
    
    def _handle_stream_response(self, response):
        """处理流式响应"""
        full_response = ""
        thoughts_container = st.empty()
        response_container = st.empty()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        data_type = data.get("type")
                        
                        if data_type == "chunk":
                            chunk = data.get("data", {}).get("text", "")
                            full_response += chunk
                            response_container.markdown(full_response)
                        
                        elif data_type == "thought":
                            thought = data.get("data", {})
                            st.session_state.agent_thoughts.append(thought)
                            self._display_thought(thought)
                        
                        elif data_type == "source":
                            source = data.get("data", {})
                            st.session_state.sources.append(source)
                            self._display_source(source)
                        
                        elif data_type == "thoughts_summary":
                            summary = data.get("data", {})
                            st.info(f"Agent思考步骤: {summary.get('count', 0)}")
                        
                        elif data_type == "error":
                            error = data.get("data", {}).get("message", "未知错误")
                            st.error(f"错误: {error}")
                            break
                        
                        elif data_type == "complete":
                            # 保存完整响应
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response
                            })
                            break
                    
                    except json.JSONDecodeError:
                        continue
        
        return full_response
    
    def _display_thought(self, thought: Dict[str, Any]):
        """显示Agent思考步骤"""
        with st.expander(f"🤔 {thought.get('step', '思考')}", expanded=False):
            cols = st.columns([1, 3])
            with cols[0]:
                st.metric("步骤", thought.get("step", ""))
            with cols[1]:
                st.text_area("思考", thought.get("thought", ""), height=100, disabled=True)
            
            if thought.get("action"):
                st.info(f"动作: {thought.get('action')}")
            if thought.get("result"):
                st.success(f"结果: {thought.get('result')}")
    
    def _display_source(self, source: Dict[str, Any]):
        """显示源文档"""
        with st.expander(f"📄 来源 {source.get('index', 0)}", expanded=False):
            st.text(f"内容: {source.get('content', '')}")
            st.text(f"元数据: {json.dumps(source.get('metadata', {}), indent=2)}")
            st.metric("相关度", f"{source.get('score', 0):.3f}")
    
    def send_message(self, message: str):
        """发送消息"""
        st.session_state.is_loading = True
        
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": message
        })
        
        # 清空思考步骤和来源
        st.session_state.agent_thoughts = []
        st.session_state.sources = []
        
        try:
            if st.session_state.streaming:
                # 流式响应
                response = self.api_client.chat_stream(
                    message=message,
                    conversation_id=st.session_state.conversation_id,
                    history=st.session_state.messages[:-1],
                    use_agent=st.session_state.use_agent,
                    **st.session_state.retrieval_config
                )
                
                self._handle_stream_response(response)
                
            else:
                # 非流式响应
                response = self.api_client.chat(
                    message=message,
                    conversation_id=st.session_state.conversation_id,
                    history=st.session_state.messages[:-1],
                    use_agent=st.session_state.use_agent,
                    **st.session_state.retrieval_config
                )
                
                # 显示响应
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.get("response", "")
                })
                
                # 显示思考步骤
                for thought in response.get("agent_thoughts", []):
                    self._display_thought(thought)
                
                # 显示来源
                for i, source in enumerate(response.get("sources", [])):
                    source["index"] = i + 1
                    self._display_source(source)
        
        except Exception as e:
            st.error(f"请求失败: {str(e)}")
        
        finally:
            st.session_state.is_loading = False
    
    def clear_chat(self):
        """清空聊天"""
        st.session_state.messages = []
        st.session_state.agent_thoughts = []
        st.session_state.sources = []
        st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        st.rerun()
    
    def run(self):
        """运行应用"""
        # 侧边栏
        with st.sidebar:
            render_sidebar(self)
        
        # 主区域
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 聊天界面
            render_chat_interface(self)
        
        with col2:
            # 配置面板
            render_config_panel(self)
        
        # 底部状态栏
        st.divider()
        col_status1, col_status2, col_status3 = st.columns(3)
        
        with col_status1:
            status = "🔴 离线" if st.session_state.is_loading else "🟢 在线"
            st.metric("状态", status)
        
        with col_status2:
            st.metric("消息数", len(st.session_state.messages))
        
        with col_status3:
            mode = "Agent模式" if st.session_state.use_agent else "简单模式"
            st.metric("模式", mode)

def main():
    app = ChatApp()
    app.run()

if __name__ == "__main__":
    main()