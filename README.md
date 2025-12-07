# Agentic-RAG-System (FastAPI + Streamlit + Chroma + LangChain)

Agentic RAG 全栈项目
核心链路：**PDF 入库 → 分块 → Embedding → Chroma 向量库 → 混合检索（关键词+向量）→ Rerank → Agent（重写/是否检索/多轮检索反思）→ SSE 流式输出 → Streamlit 逐字显示**

---

## ✨ Features（项目能力清单）

### Agentic RAG 核心能力
- **Query 重写**：提高召回质量（减少口语化/歧义，增强检索关键词）
- **检索必要性判断**：Agent 自主决定是否检索，避免“盲目调用向量库”
- **多轮检索 + 反思**：retrieve → reflect → retrieve（提升覆盖度与一致性）
- **混合检索**：关键词（BM25/关键字）+ 向量检索（Chroma）
- **重排序（Reranking）**：使用 HuggingFace **BGE reranker**
- **可观测的过程输出**：前端可展开查看“检索中/生成中/反思中”等状态与检索详情
- **SSE 流式响应**：后端 token 流式输出，前端逐字渲染（更像 ChatGPT）

### 工程能力（端到端）
- FastAPI 后端：HTTP 接口 + 业务逻辑 + RAG/Agent 编排 + JSON 返回
- Streamlit 前端：聊天 UI + 侧边栏配置 + 对话历史 + 状态栏
- ChromaDB：本地持久化向量库
- Scripts：一键初始化、入库、测试 RAG
- Git + GitHub：版本控制与可复现开发
- Docker（可选）：提供未来一致性环境与部署基础

---

## 🧱 Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit（后续替换 React / Next.js）
- **Vector DB**: ChromaDB（后续替换 Qdrant / Pinecone）
- **PDF Loader**: `UnstructuredPDFLoader`（unstructured[pdf]）
- **Embedding**: HuggingFace **BGE embedding**（如 `BAAI/bge-m3`）
- **Reranker**: HuggingFace **BGE reranker**（如 `BAAI/bge-reranker-base`）
- **LLM**: HuggingFace `meta-llama/Llama-3.1-8B-Instruct`（4-bit）
- **Agentic Orchestration**: LangChain 1.0+（现代拆包：langchain / langchain-community / langchain-huggingface 等）
- **Streaming**: SSE（Server-Sent Events）

---

## 🗂️ Project Structure

```txt
agentic-rag-system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI 主应用
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints.py # API 端点
│   │   │   └── models.py    # 数据模型
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py    # 配置文件（env）
│   │   │   └── database.py  # ChromaDB 管理
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── document_processor.py # PDF→chunk→embed→store
│   │   │   ├── retrieval_service.py  # hybrid retrieve + rerank
│   │   │   ├── agent_service.py      # agentic loop（重写/判断/多轮检索反思）
│   │   │   └── llm_service.py        # HF LLM + streaming
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── helpers.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py              # Streamlit 主应用
│   ├── components/
│   │   ├── __init__.py
│   │   ├── sidebar.py
│   │   ├── chat_interface.py
│   │   └── config_panel.py
│   └── utils/
│       └── api_client.py
├── data/                   # 文档存储（PDF/向量库等）
│   └── pdfs/
├── scripts/                # 实用脚本
│   ├── setup.py
│   ├── ingest.py
│   └── test_rag.py
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
