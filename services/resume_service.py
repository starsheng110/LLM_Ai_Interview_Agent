"""
简历解析服务
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders import PyPDFLoader

from prompts.resume_prompt import prompt_analyze_resume

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDeepSeek(model='deepseek-v4-flash', streaming=True, temperature=1.0)
    return _llm


def analyze_resume_info(resume_info) -> str:
    """
    解析上传的简历 PDF，提取结构化信息

    Args:
        resume_info: Gradio File 对象

    Returns:
        结构化的简历信息（Markdown 格式）
    """
    if resume_info is None:
        return ''
    file_path = resume_info.name

    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        resume_text = "\n".join(page.page_content for page in pages)
    except FileNotFoundError:
        return ''

    llm = _get_llm()
    chain = prompt_analyze_resume | llm | StrOutputParser()
    return chain.invoke({"resume_text": resume_text})
