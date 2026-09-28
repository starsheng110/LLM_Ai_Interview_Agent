"""
向量检索器
"""

# 面试方向 -> topic 映射
DIRECTION_TOPIC_MAP = {
    "软件测试": "软件测试面试",
    "Java工程师": "Java面试",
    "AI-Agent工程师": "AI_Agent面试"
}


def retrieve_context(vectorstore, interview_direction: str, k: int = 3) -> str:
    """
    根据面试方向从向量库中检索相关题库内容

    Args:
        vectorstore: Chroma vectorstore 实例
        interview_direction: 面试方向
        k: 返回的文档块数量

    Returns:
        拼接后的检索文本，若方向不匹配返回提示
    """
    if vectorstore is None:
        return "（题库未加载，请先构建知识库）"

    topic = DIRECTION_TOPIC_MAP.get(interview_direction, "")
    if not topic:
        return "（未找到匹配的面试方向）"

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"topic": topic}, "k": k}
    )
    docs = retriever.invoke("")
    context = "\n\n".join(doc.page_content for doc in docs)
    return context
