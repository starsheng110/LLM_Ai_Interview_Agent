"""
面试会话服务 - 封装 LangGraph 工作流调用
"""
from workflow.graph import get_interview_graph

# 默认会话 ID
_DEFAULT_SESSION_ID = "default"


def interview_conversation(
    resume_info: str,
    interview_direction: str,
    question_count: int,
    difficulty_level: str,
    user_reply: str,
    chat_history: list,
    session_id: str = _DEFAULT_SESSION_ID,
) -> tuple:
    """
    处理一轮面试对话（通过 LangGraph 工作流）

    Args:
        resume_info: 简历信息
        interview_direction: 面试方向
        question_count: 剩余题目数
        difficulty_level: 难度等级
        user_reply: 用户输入
        chat_history: 对话历史
        session_id: 会话 ID（用于 checkpointer 记忆）

    Returns:
        (清空的输入框内容, 更新后的对话历史, 剩余题数)
    """
    if not user_reply or not user_reply.strip():
        return "", chat_history, question_count

    if chat_history is None:
        chat_history = []

    graph = get_interview_graph()
    config = {"configurable": {"thread_id": session_id}}

    # 构建初始状态
    initial_state = {
        "resume_info": resume_info or "",
        "interview_direction": interview_direction or "",
        "difficulty_level": difficulty_level or "",
        "question_count": int(question_count) if question_count else 0,
        "user_reply": user_reply,
        "chat_history": chat_history,
        "retrieved_context": "",
        "current_response": "",
    }

    # 调用图
    result = graph.invoke(initial_state, config=config)

    updated_history = result.get("chat_history", chat_history)
    updated_count = result.get("question_count", question_count)

    return "", updated_history, updated_count


def clear_session(session_id: str = _DEFAULT_SESSION_ID) -> tuple:
    """
    清空当前会话

    Args:
        session_id: 会话 ID

    Returns:
        (空对话历史, 空输入, 默认题数, 默认报告文字)
    """
    graph = get_interview_graph()
    config = {"configurable": {"thread_id": session_id}}

    # 清空 checkpointer 中的状态
    checkpointer = graph.checkpointer
    if hasattr(checkpointer, 'storage'):
        checkpointer.storage.clear()

    return [], "", 4, "面试综合评估报告"
