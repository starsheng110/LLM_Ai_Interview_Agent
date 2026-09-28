"""
LangGraph 面试工作流 - 节点函数
每个节点接收 state，返回需要更新的字段
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

from prompts.interview_prompt import prompt_interview_conversation
from prompts.report_prompt import prompt_interview_report
from rag.retriever import retrieve_context

# LLM 实例（模块级单例）
_llm = None


def _get_llm():
    """获取 LLM 实例"""
    global _llm
    if _llm is None:
        _llm = ChatDeepSeek(model='deepseek-v4-flash', streaming=True, temperature=1.0)
    return _llm


# 全局 vectorstore 引用（由外部初始化后注入）
_vectorstore = None


def set_vectorstore(vectorstore):
    """注入向量库实例"""
    global _vectorstore
    _vectorstore = vectorstore


# ============ 节点函数 ============

def retrieve_node(state: dict) -> dict:
    """
    RAG 检索节点：根据面试方向从向量库检索相关题库

    Args:
        state: 当前工作流状态

    Returns:
        需要更新的状态字段
    """
    direction = state.get("interview_direction", "")
    context = retrieve_context(_vectorstore, direction, k=3)
    return {"retrieved_context": context}


def question_node(state: dict) -> dict:
    """
    出题节点：LLM 根据简历、方向、难度、题库生成面试题

    Args:
        state: 当前工作流状态

    Returns:
        需要更新的状态字段（current_response, chat_history）
    """
    llm = _get_llm()
    chain = prompt_interview_conversation | llm | StrOutputParser()

    chat_history = state.get("chat_history", [])
    response = chain.invoke({
        "resume_info": state.get("resume_info", ""),
        "interview_direction": state.get("interview_direction", ""),
        "difficulty_level": state.get("difficulty_level", ""),
        "question_count": state.get("question_count", 0),
        "retrieved_context": state.get("retrieved_context", ""),
        "user_reply": state.get("user_reply", ""),
        "chat_history": chat_history,
    })

    # 更新对话历史
    new_history = list(chat_history)
    new_history.append({"role": "user", "content": state.get("user_reply", "")})
    new_history.append({"role": "assistant", "content": response})

    return {
        "current_response": response,
        "chat_history": new_history,
    }


def evaluate_node(state: dict) -> dict:
    """
    评价节点：LLM 评价用户回答并出下一题（或结束）

    Args:
        state: 当前工作流状态

    Returns:
        需要更新的状态字段
    """
    llm = _get_llm()
    chain = prompt_interview_conversation | llm | StrOutputParser()

    # 先递减题目数
    question_count = state.get("question_count", 0)
    question_count -= 1

    chat_history = state.get("chat_history", [])
    response = chain.invoke({
        "resume_info": state.get("resume_info", ""),
        "interview_direction": state.get("interview_direction", ""),
        "difficulty_level": state.get("difficulty_level", ""),
        "question_count": question_count,
        "retrieved_context": state.get("retrieved_context", ""),
        "user_reply": state.get("user_reply", ""),
        "chat_history": chat_history,
    })

    # 更新对话历史
    new_history = list(chat_history)
    new_history.append({"role": "user", "content": state.get("user_reply", "")})
    new_history.append({"role": "assistant", "content": response})

    return {
        "current_response": response,
        "chat_history": new_history,
        "question_count": question_count,
    }


def report_node(state: dict) -> dict:
    """
    报告节点：生成面试综合评估报告

    Args:
        state: 当前工作流状态

    Returns:
        需要更新的状态字段（final_report）
    """
    llm = _get_llm()
    chain = prompt_interview_report | llm | StrOutputParser()

    chat_history = state.get("chat_history", [])
    # 格式化对话历史
    formatted_history = ""
    for msg in chat_history:
        role = '候选人' if msg.get('role') == 'user' else '面试官'
        formatted_history += f"{role}: {msg.get('content', '')}\n\n"

    final_report = chain.invoke({
        "resume_info": state.get("resume_info", ""),
        "chat_history": formatted_history,
        "interview_direction": state.get("interview_direction", ""),
        "difficulty_level": state.get("difficulty_level", ""),
    })

    return {"final_report": final_report}


def end_interview_node(state: dict) -> dict:
    """
    结束面试节点：输出结束提示（代码层面的结束，不由 LLM 判断）

    Args:
        state: 当前工作流状态

    Returns:
        需要更新的状态字段
    """
    end_msg = "面试题已全部完成。请点击下方【生成面试评估报告】按钮查看结果。"
    chat_history = list(state.get("chat_history", []))
    chat_history.append({"role": "user", "content": state.get("user_reply", "")})
    chat_history.append({"role": "assistant", "content": end_msg})

    return {
        "current_response": end_msg,
        "chat_history": chat_history,
        "question_count": 0,
    }


# ============ 条件路由函数 ============

def route_input(state: dict) -> str:
    """
    入口路由：根据用户输入判断走哪条分支

    Returns:
        下一个节点的名称
    """
    user_reply = (state.get("user_reply") or "").strip()
    question_count = state.get("question_count", 0)

    if user_reply == "开始面试":
        return "retrieve"
    elif user_reply == "结束面试":
        return "end_interview"
    elif question_count <= 0:
        return "end_interview"
    else:
        return "retrieve"


def route_after_retrieve(state: dict) -> str:
    """
    检索完成后的路由：开始面试走出题，否则走评价

    Returns:
        下一个节点的名称
    """
    user_reply = (state.get("user_reply") or "").strip()
    if user_reply == "开始面试":
        return "question"
    else:
        return "evaluate"


def route_after_evaluate(state: dict) -> str:
    """
    评价完成后的路由：还有题就回到出题（下一轮），没题就结束

    Returns:
        下一个节点的名称
    """
    question_count = state.get("question_count", 0)
    if question_count <= 0:
        return "end_interview"
    else:
        # 还有题，结束当前轮（等待用户回答下一题）
        return "__end__"
