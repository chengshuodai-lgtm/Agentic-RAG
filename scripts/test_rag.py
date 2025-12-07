#!/usr/bin/env python3
"""
RAG系统测试脚本
用于测试Agentic RAG的各项功能
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.services.agent_service import AgentService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.document_processor import DocumentProcessor

def test_retrieval():
    """测试检索功能"""
    print("🧪 测试检索功能...")
    
    retriever = RetrievalService()
    
    test_queries = [
        "什么是机器学习？",
        "深度学习有什么应用？",
        "神经网络的基本原理是什么？"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 测试混合检索
        docs, errors = retriever.hybrid_retrieval(query, top_k=3)
        
        if errors:
            print(f"  错误: {errors}")
        
        print(f"  检索到 {len(docs)} 个文档:")
        for i, doc in enumerate(docs[:2]):  # 只显示前2个
            content_preview = doc["content"][:100] + "..."
            score = doc.get("score", 0)
            print(f"    {i+1}. {content_preview} (分数: {score:.3f})")
        
        # 测试检索判断
        need_retrieval, reason = retriever.judge_retrieval_need(query, [])
        print(f"  需要检索: {need_retrieval} ({reason})")
        
        # 测试查询重写
        rewritten = retriever.rewrite_query(query, [])
        print(f"  重写后: {rewritten}")
    
    print("\n✅ 检索功能测试完成")

def test_agent():
    """测试Agent功能"""
    print("\n🧪 测试Agent功能...")
    
    agent = AgentService()
    
    test_cases = [
        {
            "query": "请解释一下机器学习的基本概念",
            "use_agent": True
        },
        {
            "query": "你好，今天天气怎么样？",
            "use_agent": False
        },
        {
            "query": "基于文档内容，总结深度学习的核心思想",
            "use_agent": True
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        print(f"  查询: {test_case['query']}")
        print(f"  Agent模式: {test_case['use_agent']}")
        
        start_time = time.time()
        
        try:
            for response in agent.process_query(
                query=test_case["query"],
                conversation_id=f"test_{i}",
                use_agent=test_case["use_agent"],
                stream=False
            ):
                if response["type"] == "complete":
                    data = response["data"]
                    print(f"  响应长度: {len(data['response'])} 字符")
                    print(f"  思考步骤: {len(data.get('thoughts', []))} 个")
                    print(f"  参考文档: {len(data.get('sources', []))} 个")
                    
                    # 显示前2个思考步骤
                    thoughts = data.get("thoughts", [])
                    for j, thought in enumerate(thoughts[:2]):
                        print(f"    思考{j+1}: {thought.get('step')} - {thought.get('thought')[:50]}...")
                    
                elif response["type"] == "error":
                    print(f"  ❌ 错误: {response['data']['error']}")
        
        except Exception as e:
            print(f"  ❌ 异常: {e}")
        
        print(f"  耗时: {time.time() - start_time:.2f}秒")
    
    print("\n✅ Agent功能测试完成")

def test_document_processing():
    """测试文档处理功能"""
    print("\n🧪 测试文档处理功能...")
    
    processor = DocumentProcessor()
    
    # 创建一个测试PDF（如果没有的话）
    test_pdf = Path("data/test/test.pdf")
    if not test_pdf.exists():
        print("⚠️  测试PDF不存在，跳过文档处理测试")
        print("   请将测试PDF放在 data/test/test.pdf")
        return
    
    print(f"处理测试文件: {test_pdf}")
    
    result = processor.process_and_store(str(test_pdf), "test_collection")
    
    if result["status"] == "success":
        print(f"✅ 文档处理成功")
        print(f"   块数量: {result['total_chunks']}")
        print(f"   集合: {result['collection']}")
        print(f"   总文档: {result['total_documents']}")
    else:
        print(f"❌ 文档处理失败: {result.get('error', '未知错误')}")
    
    # 获取集合信息
    info = processor.get_collection_info("test_collection")
    print(f"\n集合信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
    
    print("\n✅ 文档处理测试完成")

def performance_test():
    """性能测试"""
    print("\n📊 性能测试...")
    
    agent = AgentService()
    
    # 简单查询
    simple_queries = [
        "你好",
        "介绍一下你自己",
        "今天天气怎么样"
    ]
    
    # 复杂查询
    complex_queries = [
        "详细解释机器学习中的监督学习和无监督学习的区别",
        "基于文档内容，说明深度学习在图像识别中的应用",
        "总结神经网络训练的基本步骤和注意事项"
    ]
    
    def run_query_set(name, queries, use_agent):
        print(f"\n{name}查询测试 ({'Agent' if use_agent else '简单'}模式):")
        
        times = []
        for i, query in enumerate(queries):
            start_time = time.time()
            
            try:
                for response in agent.process_query(
                    query=query,
                    conversation_id=f"perf_{name}_{i}",
                    use_agent=use_agent,
                    stream=False
                ):
                    if response["type"] in ["complete", "error"]:
                        break
            
            except Exception as e:
                print(f"  ❌ 查询失败: {e}")
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  查询{i+1}: {elapsed:.2f}秒")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"  平均耗时: {avg_time:.2f}秒")
            print(f"  最快: {min(times):.2f}秒")
            print(f"  最慢: {max(times):.2f}秒")
    
    run_query_set("简单", simple_queries, use_agent=False)
    run_query_set("简单", simple_queries, use_agent=True)
    run_query_set("复杂", complex_queries, use_agent=False)
    run_query_set("复杂", complex_queries, use_agent=True)
    
    print("\n✅ 性能测试完成")

def main():
    """主测试函数"""
    print("🚀 开始Agentic RAG系统测试")
    print("=" * 50)
    
    try:
        # 测试检索功能
        test_retrieval()
        
        # 测试文档处理功能
        test_document_processing()
        
        # 测试Agent功能
        test_agent()
        
        # 性能测试
        performance_test()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()