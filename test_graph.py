"""测试 LangGraph 工作流基本功能"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflow.graph import build_interview_graph
from workflow.nodes import set_vectorstore

# 不加载真实向量库，用 None 代替（retrieve 节点会返回提示文字）
set_vectorstore(None)

graph = build_interview_graph()
config = {"configurable": {"thread_id": "test_001"}}

print("=" * 50)
print("测试 1：开始面试（出题）")
print("=" * 50)

state = {
    "resume_info": "测试候选人，Java 开发，3 年经验",
    "interview_direction": "Java工程师",
    "difficulty_level": "中",
    "question_count": 3,
    "user_reply": "开始面试",
    "chat_history": [],
}

result = graph.invoke(state, config=config)
print(f"剩余题数: {result.get('question_count')}")
print(f"对话历史条数: {len(result.get('chat_history', []))}")
print(f"LLM 回复前50字: {result.get('current_response', '')[:50]}...")

print()
print("=" * 50)
print("测试 2：回答问题（评价+出下一题）")
print("=" * 50)

state2 = {
    "resume_info": "测试候选人，Java 开发，3 年经验",
    "interview_direction": "Java工程师",
    "difficulty_level": "中",
    "question_count": result.get("question_count", 3),
    "user_reply": "我对 Java 并发编程有一定了解，使用过 synchronized 和 ReentrantLock...",
    "chat_history": result.get("chat_history", []),
}

result2 = graph.invoke(state2, config=config)
print(f"剩余题数: {result2.get('question_count')}")
print(f"对话历史条数: {len(result2.get('chat_history', []))}")
print(f"LLM 回复前50字: {result2.get('current_response', '')[:50]}...")

print()
print("=" * 50)
print("🎉 LangGraph 工作流运行成功！")
print("=" * 50)
