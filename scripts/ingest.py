#!/usr/bin/env python3
"""
文档处理脚本
用于批量处理PDF文档并导入到向量数据库
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Any
import time

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.services.document_processor import DocumentProcessor
from backend.app.core.config import settings
from backend.app.core.database import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_directory(input_dir: str, collection_name: str = "default") -> List[Dict[str, Any]]:
    """处理目录中的所有PDF文档"""
    results = []
    doc_processor = DocumentProcessor()
    
    # 获取所有PDF文件
    pdf_files = []
    for ext in ['pdf', 'PDF']:
        pdf_files.extend(list(Path(input_dir).glob(f"**/*.{ext}")))
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"处理文件: {pdf_file}")
            start_time = time.time()
            
            result = doc_processor.process_and_store(
                str(pdf_file), 
                collection_name
            )
            
            result["file"] = str(pdf_file)
            result["processing_time"] = time.time() - start_time
            
            if result["status"] == "success":
                logger.info(f"成功处理: {pdf_file.name} -> {result['total_chunks']}个块")
            else:
                logger.error(f"处理失败: {pdf_file.name} -> {result.get('error', '未知错误')}")
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"处理文件时出错 {pdf_file}: {e}")
            results.append({
                "status": "error",
                "file": str(pdf_file),
                "error": str(e)
            })
    
    return results

def show_collection_info(collection_name: str = "default"):
    """显示集合信息"""
    try:
        doc_processor = DocumentProcessor()
        info = doc_processor.get_collection_info(collection_name)
        
        print(f"\n{'='*50}")
        print(f"集合: {info.get('collection', collection_name)}")
        print(f"文档数量: {info.get('total_documents', 0)}")
        
        if "metadata" in info:
            print("元数据:")
            for key, value in info["metadata"].items():
                print(f"  {key}: {value}")
        
        print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"获取集合信息失败: {e}")

def clear_collection(collection_name: str = "default"):
    """清空集合"""
    try:
        from chromadb import PersistentClient
        from chromadb.config import Settings
        
        client = PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        client.delete_collection(name=collection_name)
        logger.info(f"集合 '{collection_name}' 已删除")
        
    except Exception as e:
        logger.error(f"清空集合失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="文档处理脚本")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # ingest命令
    ingest_parser = subparsers.add_parser("ingest", help="处理文档")
    ingest_parser.add_argument("--input", "-i", required=True, help="输入目录或文件")
    ingest_parser.add_argument("--collection", "-c", default="default", help="集合名称")
    ingest_parser.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录")
    
    # info命令
    info_parser = subparsers.add_parser("info", help="查看集合信息")
    info_parser.add_argument("--collection", "-c", default="default", help="集合名称")
    
    # clear命令
    clear_parser = subparsers.add_parser("clear", help="清空集合")
    clear_parser.add_argument("--collection", "-c", default="default", help="集合名称")
    clear_parser.add_argument("--confirm", action="store_true", help="确认删除")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        input_path = Path(args.input)
        
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            # 处理单个文件
            doc_processor = DocumentProcessor()
            result = doc_processor.process_and_store(str(input_path), args.collection)
            
            if result["status"] == "success":
                print(f"✅ 成功处理: {input_path.name}")
                print(f"   块数量: {result['total_chunks']}")
                print(f"   总文档: {result['total_documents']}")
            else:
                print(f"❌ 处理失败: {result.get('error', '未知错误')}")
                
        elif input_path.is_dir():
            # 处理目录
            results = process_directory(str(input_path), args.collection)
            
            # 统计结果
            success_count = sum(1 for r in results if r["status"] == "success")
            total_chunks = sum(r.get("total_chunks", 0) for r in results if r["status"] == "success")
            
            print(f"\n{'='*50}")
            print(f"处理完成!")
            print(f"成功: {success_count}/{len(results)} 个文件")
            print(f"总块数: {total_chunks}")
            print(f"{'='*50}")
            
        else:
            print(f"❌ 无效的输入路径: {args.input}")
    
    elif args.command == "info":
        show_collection_info(args.collection)
    
    elif args.command == "clear":
        if args.confirm:
            clear_collection(args.collection)
        else:
            print("⚠️  危险操作! 使用 --confirm 参数确认删除集合")
            print(f"   将会删除集合: {args.collection}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()