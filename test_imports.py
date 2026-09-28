"""快速验证脚本 - 测试各层 import 和基本功能"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("1. 测试 prompts 层 import...")
from prompts.resume_prompt import prompt_analyze_resume
from prompts.interview_prompt import prompt_interview_conversation
from prompts.report_prompt import prompt_interview_report
print("   ✓ prompts 层导入成功")

print("=" * 50)
print("2. 测试 rag 层 import...")
from rag.retriever import retrieve_context, DIRECTION_TOPIC_MAP
print("   ✓ rag 层导入成功")
print(f"   方向映射: {DIRECTION_TOPIC_MAP}")

print("=" * 50)
print("3. 测试 workflow 层 import...")
from workflow.state import InterviewState
from workflow.graph import build_interview_graph
print("   ✓ workflow 层导入成功")

# 构建图（不加载向量库，不启动 LLM）
graph = build_interview_graph()
print("   ✓ LangGraph 图构建成功")
print(f"   图节点: {list(graph.get_graph().nodes.keys())}")

print("=" * 50)
print("4. 测试 services 层 import...")
from services.resume_service import analyze_resume_info
from services.interview_service import interview_conversation, clear_session
from services.report_service import interview_report
print("   ✓ services 层导入成功")

print("=" * 50)
print("5. 测试 ui 层 import...")
from ui.gradio_app import build_demo
print("   ✓ ui 层导入成功")

print("=" * 50)
print("🎉 所有层 import 测试通过！")
print("=" * 50)
