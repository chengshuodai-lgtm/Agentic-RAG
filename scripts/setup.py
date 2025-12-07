#!/usr/bin/env python3
"""
项目设置脚本
初始化项目环境、下载模型等
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging
import platform
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境"""
    print("🔍 检查环境...")
    
    checks = {
        "Python版本": sys.version_info >= (3, 10),
        "操作系统": platform.system() in ["Darwin", "Linux"],
        "内存": True,  # 简化检查
        "磁盘空间": True,
    }
    
    all_ok = True
    for check_name, check_result in checks.items():
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
        if not check_result:
            all_ok = False
    
    return all_ok

def setup_conda_env():
    """设置Conda环境"""
    env_name = "agentic-rag"
    
    print(f"\n🐍 设置Conda环境 '{env_name}'...")
    
    # 检查是否已存在
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True
    )
    
    if env_name in result.stdout:
        print(f"  ⚠️  环境 '{env_name}' 已存在")
        response = input("  是否重新创建? (y/N): ")
        if response.lower() != 'y':
            print("  ✅ 使用现有环境")
            return True
    
    # 创建环境
    print("  创建Conda环境...")
    cmd = [
        "conda", "create", "-n", env_name,
        "python=3.10", "-y"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("  ✅ Conda环境创建成功")
        return True
    else:
        print(f"  ❌ Conda环境创建失败: {result.stderr}")
        return False

def install_dependencies():
    """安装依赖"""
    print("\n📦 安装依赖...")
    
    # 后端依赖
    print("  安装后端依赖...")
    backend_reqs = [
        "langchain==0.1.0",
        "fastapi", "uvicorn[standard]", 
        "streamlit", "chromadb", "pypdf", "unstructured",
        "sentence-transformers", "FlagEmbedding",
        "transformers", "accelerate", "bitsandbytes",
        "torch", "torchvision", "torchaudio",
        "pydantic-settings", "python-dotenv",
        "langchain-community", "langchain-core", "langchain-text-splitters",
        "tiktoken", "einops", "requests", "websockets"
    ]
    
    for package in backend_reqs:
        print(f"   安装 {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"    ⚠️  {package} 安装失败: {result.stderr[:100]}")
    
    print("  ✅ 依赖安装完成")

def download_models():
    """下载模型"""
    print("\n🤖 下载模型...")
    
    models = {
        "embedding": "BAAI/bge-large-zh-v1.5",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "llm": "meta-llama/Llama-3.1-8B-Instruct"
    }
    
    # 创建模型目录
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # 检查是否需要HF Token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token and "meta-llama" in models["llm"]:
        print("  ⚠️  检测到Llama模型，需要HuggingFace Token")
        token = input("  请输入HF_TOKEN (或按回车跳过): ")
        if token:
            os.environ["HF_TOKEN"] = token
            with open(".env", "a") as f:
                f.write(f"\nHF_TOKEN={token}\n")
    
    # 简化模型下载（实际使用时会自动下载）
    print("  📝 模型将在首次使用时自动下载")
    print("  💡 提示: 确保有足够的磁盘空间(约20GB)")
    
    return True

def setup_project_structure():
    """设置项目结构"""
    print("\n📁 设置项目结构...")
    
    directories = [
        "backend/app/api",
        "backend/app/core", 
        "backend/app/services",
        "backend/app/utils",
        "frontend/components",
        "frontend/utils",
        "data/pdfs",
        "data/uploads",
        "chroma_db",
        "models",
        "logs",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  创建目录: {directory}")
    
    # 创建示例配置文件
    env_example = """# 项目配置
PROJECT_NAME=Agentic RAG System
VERSION=1.0.0

# HuggingFace配置
HF_TOKEN=your_huggingface_token_here

# 模型配置
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

# 服务器配置
HOST=0.0.0.0
PORT=8000
STREAMLIT_PORT=8501

# 检索配置
RETRIEVAL_TOP_K=10
RERANK_TOP_K=5
SIMILARITY_THRESHOLD=0.7

# 分块配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    print("  创建文件: .env.example")
    
    # 复制示例配置
    if not Path(".env").exists():
        shutil.copy(".env.example", ".env")
        print("  创建文件: .env (请修改配置)")
    
    print("  ✅ 项目结构设置完成")

def create_requirements():
    """创建requirements.txt文件"""
    print("\n📄 创建requirements.txt...")
    
    # 获取已安装的包
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        with open("backend/requirements.txt", "w") as f:
            f.write(result.stdout)
        print("  ✅ requirements.txt 创建完成")
    else:
        print("  ⚠️  无法创建requirements.txt")

def setup_git():
    """设置Git"""
    print("\n🔧 设置Git版本控制...")
    
    if not Path(".git").exists():
        # 初始化Git
        subprocess.run(["git", "init"], capture_output=True)
        print("  ✅ Git仓库初始化")
    
    # 创建.gitignore
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project
chroma_db/
data/uploads/
models/
logs/
temp/
*.pdf
*.log

# Streamlit
.streamlit/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore)
    print("  ✅ .gitignore 创建完成")

def show_next_steps():
    """显示下一步"""
    print("\n" + "="*50)
    print("🎉 项目设置完成!")
    print("="*50)
    
    print("\n📋 下一步:")
    print("  1. 修改 .env 文件中的配置")
    print("  2. 准备一些PDF文档到 data/pdfs/ 目录")
    print("  3. 激活Conda环境:")
    print("     $ conda activate agentic-rag")
    print("  4. 处理文档:")
    print("     $ python scripts/ingest.py --input data/pdfs/")
    print("  5. 启动后端服务:")
    print("     $ cd backend && python -m app.main")
    print("  6. 启动前端服务:")
    print("     $ streamlit run frontend/app.py")
    print("\n🌐 访问地址:")
    print("  - 前端: http://localhost:8501")
    print("  - 后端API: http://localhost:8000")
    print("  - API文档: http://localhost:8000/docs")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="项目设置脚本")
    parser.add_argument("--skip-env", action="store_true", help="跳过环境检查")
    parser.add_argument("--skip-models", action="store_true", help="跳过模型下载")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖安装")
    
    args = parser.parse_args()
    
    print("🚀 Agentic RAG 项目设置")
    print("="*50)
    
    try:
        # 检查环境
        if not args.skip_env and not check_environment():
            print("\n❌ 环境检查失败，请解决问题后重试")
            return
        
        # 设置Conda环境
        if not setup_conda_env():
            print("\n❌ Conda环境设置失败")
            return
        
        # 设置项目结构
        setup_project_structure()
        
        # 安装依赖
        if not args.skip_deps:
            install_dependencies()
        
        # 下载模型
        if not args.skip_models:
            download_models()
        
        # 创建requirements.txt
        create_requirements()
        
        # 设置Git
        setup_git()
        
        # 显示下一步
        show_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  设置被用户中断")
    except Exception as e:
        print(f"\n❌ 设置过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()