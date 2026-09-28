"""
LangGraph 面试工作流 - 图构建
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from workflow.state import InterviewState
from workflow.nodes import (
    retrieve_node,
    question_node,
    evaluate_node,
    report_node,
    end_interview_node,
    route_input,
    route_after_retrieve,
    route_after_evaluate,
)


def build_interview_graph(checkpointer=None):
    """
    构建面试工作流图

    流程：
    START → route_input（条件路由）
      ├─ "开始面试" → retrieve → question → END
      ├─ "结束面试" / count<=0 → end_interview → END
      └─ 正常回答 → retrieve → evaluate → route_after_evaluate
                                                    ├─ count<=0 → end_interview → END
                                                    └─ count>0 → END（等待下轮）

    Args:
        checkpointer: 状态持久化检查点，None 则使用 MemorySaver

    Returns:
        编译后的 LangGraph 图实例
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = StateGraph(InterviewState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("question", question_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("report", report_node)
    workflow.add_node("end_interview", end_interview_node)

    # 设置入口（条件路由）
    workflow.set_conditional_entry_point(
        route_input,
        {
            "retrieve": "retrieve",
            "end_interview": "end_interview",
        }
    )

    # retrieve 完成后，根据是否"开始面试"决定出题还是评价
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "question": "question",
            "evaluate": "evaluate",
        }
    )

    # question 出题完成，结束当前轮
    workflow.add_edge("question", END)

    # evaluate 评价完成后，判断是否还有题
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "end_interview": "end_interview",
            "__end__": END,
        }
    )

    # end_interview 结束面试
    workflow.add_edge("end_interview", END)

    # report 生成报告后结束
    workflow.add_edge("report", END)

    # 编译图（带 checkpointer）
    graph = workflow.compile(checkpointer=checkpointer)
    return graph


# 全局图实例
_graph = None


def get_interview_graph(checkpointer=None):
    """
    单例获取面试图实例

    Args:
        checkpointer: 状态持久化检查点

    Returns:
        编译后的 LangGraph 图实例
    """
    global _graph
    if _graph is None:
        _graph = build_interview_graph(checkpointer)
    return _graph
