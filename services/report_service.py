"""
面试报告生成服务
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

from prompts.report_prompt import prompt_interview_report

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDeepSeek(model='deepseek-v4-flash', streaming=True, temperature=1.0)
    return _llm


def interview_report(
    resume_info: str,
    chat_history: list,
    interview_direction: str,
    difficulty_level: str,
    session_id: str = "default",
) -> str:
    """
    生成面试综合评估报告

    Args:
        resume_info: 简历信息
        chat_history: 对话历史
        interview_direction: 面试方向
        difficulty_level: 难度等级
        session_id: 会话 ID（预留）

    Returns:
        Markdown 格式的面试评估报告
    """
    if not chat_history or len(chat_history) == 0:
        return "⚠️ 当前没有有效的面试记录，请先完成面试对话后再生成报告。"

    # 格式化对话历史
    formatted_history = ""
    for msg in chat_history:
        role = '候选人' if msg.get('role') == 'user' else '面试官'
        formatted_history += f"{role}: {msg.get('content', '')}\n\n"

    llm = _get_llm()
    chain = prompt_interview_report | llm | StrOutputParser()
    final_report = chain.invoke({
        "resume_info": resume_info,
        "chat_history": formatted_history,
        "interview_direction": interview_direction,
        "difficulty_level": difficulty_level,
    })
    return final_report
