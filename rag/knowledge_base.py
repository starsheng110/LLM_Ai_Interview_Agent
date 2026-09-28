"""
知识库构建与管理
"""
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from rag.embeddings import get_embeddings


def _get_topic_by_filename(source_path: str) -> str:
    """根据文件名判断所属面试方向的 topic"""
    path_lower = source_path.lower()
    if 'java' in path_lower:
        return 'Java面试'
    elif 'ai-agent' in path_lower:
        return 'AI_Agent面试'
    elif '软件测试' in path_lower:
        return '软件测试面试'
    return ''


def build_knowledge_base(docs_dir: list, db_dir: str = "./chroma_db"):
    """
    构建向量知识库：加载文档 -> 分块 -> 打 topic 标签 -> 向量化存入 Chroma

    Args:
        docs_dir: 文档路径列表
        db_dir: Chroma 持久化目录

    Returns:
        Chroma vectorstore 实例
    """
    docs = []
    for path in docs_dir:
        ext = os.path.splitext(path)[1].lstrip('.')
        if ext == "pdf":
            loader = PyPDFLoader(path)
        elif ext == "docx":
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())
    print(f"加载了 {len(docs)} 个文件")

    # 按文件名打 topic 标签
    for doc in docs:
        source_path = doc.metadata.get("source", "").lower()
        topic = _get_topic_by_filename(source_path)
        if topic:
            doc.metadata['topic'] = topic

    # 分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"分割成 {len(chunks)} 个块")

    # 向量化存入 Chroma
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_dir
    )
    print(f"向量数据库已存入: {db_dir}")
    return vectorstore


def load_knowledge_base(db_dir: str = "./chroma_db"):
    """
    加载已有的向量知识库

    Args:
        db_dir: Chroma 持久化目录

    Returns:
        Chroma vectorstore 实例
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )
    print(f"已加载现有向量数据库: {db_dir}")
    return vectorstore


def init_vectorstore(db_dir: str = "./chroma_db", docs_dir: list = None):
    """
    初始化向量库：存在则加载，不存在则构建

    Args:
        db_dir: Chroma 持久化目录
        docs_dir: 文档路径列表（首次构建时需要）

    Returns:
        Chroma vectorstore 实例
    """
    if os.path.exists(db_dir):
        return load_knowledge_base(db_dir)
    else:
        if docs_dir is None:
            raise ValueError("首次构建知识库需要提供 docs_dir 参数")
        return build_knowledge_base(docs_dir, db_dir=db_dir)
