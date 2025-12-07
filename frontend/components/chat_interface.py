import streamlit as st
import json

def render_chat_interface(app):
    """渲染聊天界面"""
    st.title("💬 Agentic RAG Chat")
    
    # 显示聊天消息
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 显示消息索引
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.caption(f"#{i+1}")
    
    # 显示Agent思考过程
    if st.session_state.agent_thoughts:
        with st.expander("🤔 Agent思考过程", expanded=True):
            for thought in st.session_state.agent_thoughts[-5:]:  # 显示最近5个思考
                app._display_thought(thought)
    
    # 显示检索来源
    if st.session_state.sources:
        with st.expander("📄 检索来源", expanded=False):
            for source in st.session_state.sources[:3]:  # 显示前3个来源
                app._display_source(source)
    
    # 输入区域
    st.divider()
    
    col_input1, col_input2 = st.columns([5, 1])
    
    with col_input1:
        user_input = st.chat_input(
            "输入您的问题...",
            key="user_input",
            disabled=st.session_state.is_loading
        )
    
    with col_input2:
        if st.button("🔄", help="重新生成", disabled=st.session_state.is_loading or not st.session_state.messages):
            if st.session_state.messages:
                last_user_msg = None
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        last_user_msg = msg["content"]
                        break
                
                if last_user_msg:
                    # 移除之前的assistant回复
                    if st.session_state.messages[-1]["role"] == "assistant":
                        st.session_state.messages.pop()
                    
                    # 重新发送
                    app.send_message(last_user_msg)
    
    # 处理用户输入
    if user_input:
        app.send_message(user_input)
        st.rerun()
    
    # 加载状态
    if st.session_state.is_loading:
        with st.status("Agent正在思考...", expanded=True) as status:
            st.write("🔍 分析查询...")
            st.write("📚 检索文档...")
            st.write("🤔 整合信息...")
            st.write("✍️ 生成回答...")