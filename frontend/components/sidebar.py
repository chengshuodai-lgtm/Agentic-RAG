import streamlit as st
import os
import requests
import json

def render_sidebar(app):
    """渲染侧边栏"""
    st.title("🤖 Agentic RAG")
    
    st.divider()
    
    # 对话管理
    st.subheader("对话管理")
    
    if st.button("🆕 新对话", use_container_width=True):
        app.clear_chat()
    
    if st.button("🗑️ 清空历史", use_container_width=True):
        if st.session_state.conversation_id:
            try:
                response = app.api_client.delete_conversation(
                    st.session_state.conversation_id
                )
                if response.get("status") == "success":
                    st.success("对话历史已删除")
                    app.clear_chat()
            except Exception as e:
                st.error(f"删除失败: {e}")
    
    st.divider()
    
    # 文档管理
    st.subheader("文档管理")
    
    uploaded_file = st.file_uploader(
        "上传PDF文档",
        type=["pdf"],
        help="上传PDF文件到知识库"
    )
    
    if uploaded_file is not None:
        # 保存上传的文件
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("📤 处理文档", use_container_width=True):
            with st.spinner("处理文档中..."):
                try:
                    response = app.api_client.upload_document(file_path)
                    if response.get("status") == "success":
                        st.success(f"处理完成: {response.get('total_chunks')}个块")
                    else:
                        st.error(f"处理失败: {response.get('message')}")
                except Exception as e:
                    st.error(f"上传失败: {e}")
    
    # 文档集合信息
    if st.button("📊 查看集合", use_container_width=True):
        try:
            response = app.api_client.get_collection_info()
            st.json(response)
        except Exception as e:
            st.error(f"获取失败: {e}")
    
    st.divider()
    
    # 设置
    st.subheader("设置")
    
    # 模式选择
    mode = st.radio(
        "模式选择",
        ["🤖 Agent模式", "⚡ 简单模式"],
        index=0 if st.session_state.use_agent else 1,
        help="Agent模式包含查询重写、多轮检索等高级功能"
    )
    st.session_state.use_agent = mode == "🤖 Agent模式"
    
    # 响应方式
    st.session_state.streaming = st.toggle(
        "流式响应",
        value=True,
        help="启用流式响应可以获得更好的交互体验"
    )
    
    st.divider()
    
    # 系统信息
    st.subheader("系统信息")
    
    # 健康检查
    if st.button("🩺 健康检查", use_container_width=True):
        try:
            response = app.api_client.health_check()
            st.success(f"状态: {response.get('status', 'unknown')}")
        except Exception as e:
            st.error(f"服务不可用: {e}")
    
    # 版本信息
    st.caption("Agentic RAG System v1.0")
    st.caption("基于LangChain + FastAPI + Streamlit")