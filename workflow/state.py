"""
LangGraph 面试工作流 - 状态定义
"""
from typing import TypedDict, List, Dict


class InterviewState(TypedDict, total=False):
    """
    面试工作流的全局状态
    所有节点共享并修改此状态
    """
    # 基础配置
    resume_info: str              # 简历信息
    interview_direction: str      # 面试方向
    difficulty_level: str         # 难度等级
    question_count: int           # 剩余题目数

    # 交互数据
    user_reply: str               # 用户当前输入
    chat_history: List[Dict]      # 对话历史 [{role, content}, ...]

    # RAG 结果
    retrieved_context: str        # 检索到的题库内容

    # 输出结果
    current_response: str        # 当前轮 LLM 输出
    final_report: str             # 最终面试报告
