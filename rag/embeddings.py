"""
Embedding 模型封装
"""
from langchain_community.embeddings import HuggingFaceEmbeddings

# BGE 中文嵌入模型路径
_EMBEDDING_MODEL_PATH = r"D:\modelbig\models--BAAI--bge-large-zh-v1.5\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116"

_embeddings = None


def get_embeddings():
    """单例获取 embedding 模型实例"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings
