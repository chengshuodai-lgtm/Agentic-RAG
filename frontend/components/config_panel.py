import streamlit as st

def render_config_panel(app):
    """渲染配置面板"""
    st.header("⚙️ 配置")
    
    # 检索配置
    with st.expander("🔍 检索设置", expanded=True):
        st.session_state.retrieval_config["top_k"] = st.slider(
            "检索数量 (top_k)",
            min_value=1,
            max_value=20,
            value=5,
            help="每次检索返回的文档数量"
        )
        
        st.session_state.retrieval_config["use_reranker"] = st.toggle(
            "启用重排序",
            value=True,
            help="启用BGE重排序模型提高相关性"
        )
    
    # 生成配置
    with st.expander("🧠 生成设置", expanded=True):
        st.session_state.retrieval_config["temperature"] = st.slider(
            "温度 (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="控制生成随机性，越低越确定"
        )
        
        max_tokens = st.slider(
            "最大生成长度",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="生成回答的最大长度"
        )
        st.session_state.retrieval_config["max_tokens"] = max_tokens
    
    # Agent配置
    if st.session_state.use_agent:
        with st.expander("🤖 Agent设置", expanded=True):
            enable_rewrite = st.toggle(
                "查询重写",
                value=True,
                help="启用查询重写优化检索"
            )
            st.session_state.retrieval_config["enable_rewrite"] = enable_rewrite
            
            enable_judge = st.toggle(
                "检索判断",
                value=True,
                help="智能判断是否需要检索"
            )
            st.session_state.retrieval_config["enable_judge"] = enable_judge
            
            max_turns = st.slider(
                "最大检索轮数",
                min_value=1,
                max_value=5,
                value=3,
                help="多轮检索的最大次数"
            )
            st.session_state.retrieval_config["max_turns"] = max_turns
    
    # 测试查询
    with st.expander("🚀 测试查询", expanded=False):
        test_queries = [
            "什么是机器学习？",
            "解释深度学习的基本概念",
            "如何构建一个神经网络？",
            "机器学习有哪些应用场景？"
        ]
        
        for query in test_queries:
            if st.button(query, use_container_width=True, key=f"test_{query}"):
                st.session_state.user_input = query
    
    # 系统状态
    with st.expander("📊 系统状态", expanded=False):
        st.metric("对话ID", st.session_state.conversation_id[:8])
        st.metric("历史消息", len(st.session_state.messages))
        st.metric("思考步骤", len(st.session_state.agent_thoughts))
        st.metric("检索来源", len(st.session_state.sources))