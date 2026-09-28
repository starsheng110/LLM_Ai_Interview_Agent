## Ch05 LangChain实战-RAG

### 5. RAG（检索增强生成）：让 LLM 基于你的数据回答

RAG 是当前最热门的 LLM 应用模式——你的公司有几百份内部文档，想让 AI 基于这些文档回答问题，但又不想花大钱微调模型。**RAG 就是答案**：先检索相关文档片段，再把它们塞进 prompt 让 LLM 回答。

#### 5.1 RAG 是什么？为什么比微调更实用？

```tex
RAG（Retrieval-Augmented Generation）= 检索 + 生成

传统方式：用户提问 → LLM 凭"记忆"回答 → 可能瞎编（幻觉）
RAG方式：用户提问 → 检索相关文档 → 把文档+问题一起给LLM → 基于事实回答
```

**RAG vs 微调 vs 长上下文：**

| 方案         | 成本                 | 实时性                     | 适用场景               |
| :----------- | :------------------- | :------------------------- | :--------------------- |
| **RAG**      | 低（只需 Embedding） | 高（文档更新即生效）       | 知识库问答、文档助手   |
| **微调**     | 高（需要 GPU 训练）  | 低（每次更新都要重新训练） | 风格定制、领域专用模型 |
| **长上下文** | 中（token 费用高）   | 高                         | 少量文档，一次性使用   |

**RAG 的完整流程：**

```
离线阶段（只做一次）：
文档 → 分割成小块 → 向量化（Embedding）→ 存入向量数据库

在线阶段（每次提问）：
用户问题 → 向量化 → 在向量数据库中搜索相似文档块 → 把相关文档块 + 问题拼成 prompt → LLM 回答
```

#### 5.2 文档加载：PDF、Markdown、网页

LangChain 提供了几十种文档加载器，覆盖常见格式：

```python
# ── 加载 Markdown 文件 ──
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./docs/guide.md", encoding="utf-8")
docs = loader.load()
print(f"加载了 {len(docs)} 个文档")
print(docs[0].page_content[:200])  # 文档内容
print(docs[0].metadata)             # {'source': './docs/guide.md'}

# ── 加载 PDF ──
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./docs/report.pdf")
docs = loader.load()  # 每页一个 Document 对象
print(f"共 {len(docs)} 页")

# ── 加载网页 ──
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.python.org/3/tutorial/index.html")
docs = loader.load()

# ── 批量加载目录下的所有文件 ──
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./docs/", glob="**/*.md", show_progress=True)
docs = loader.load()
print(f"加载了 {len(docs)} 个文档")
	
```

> 💡 每个 `Document` 对象包含两个属性：`page_content`（文本内容）和 `metadata`（来源信息，如文件路径、页码）。metadata 在最终回答时可以用来标注"信息来源"。

#### 5.3 文本分割：RecursiveCharacterTextSplitter

一份文档可能有几万字，但 LLM 的 context window 有限（而且塞太多内容效果也不好）。所以需要把文档**切成小块**：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每块最多500个字符
    chunk_overlap=50,     # 相邻块重叠50个字符（防止语义被截断）
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 分割优先级
)

# 分割文档
chunks = splitter.split_documents(docs)
print(f"原始文档 {len(docs)} 个 → 分割成 {len(chunks)} 个块")
print(f"每块大约 {len(chunks[0].page_content)} 个字符")
```

**分割策略的关键参数：**

```
chunk_size=500, chunk_overlap=50 的效果：

原文：[████████████████████████████████████████████]
块 1：[██████████████]
块 2：        [██████████████]      ← 有 50 字符重叠
块 3：                [██████████████]
```

| 参数            | 建议值       | 说明                                 |
| :-------------- | :----------- | :----------------------------------- |
| `chunk_size`    | 300~1000     | 太小失去上下文，太大检索不精准       |
| `chunk_overlap` | 50~100       | 防止关键信息在分割边界被截断         |
| `separators`    | 按语义优先级 | 优先按段落分，其次按句子，最后按字符 |

> 💡 **分割质量直接影响 RAG 效果**。建议根据你的文档类型调整参数——代码文档用较小的 chunk_size（300），叙事性文档用较大的（800~1000）。

#### 5.4 向量存储：Embedding + Chroma/FAISS

分割好的文本块需要转成**向量（数字数组）**，才能做语义搜索。这个过程叫 Embedding。

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# — Step 1: 创建 Embedding 模型（第一次运行时会下载该模型） — 
'''embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)'''

# -Step 1: 定义 embedings
persist_dir = "./chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="model/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# — Step 2: 把文档块存入向量数据库 —
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    # 向量库已存在，直接加载
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print("已加载已有向量数据库")
else:
    # 首次运行，生成向量库
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"已存入{vectorstore._collection.count()}个向量")

# — Step 3: 语义搜索 —
results = vectorstore.similarity_search("什么是装饰器？", k=3)
for doc in results:
    print(f"📄 来源: {doc.metadata.get('source', '未知')}")
    print(f"📝 内容: {doc.page_content[:100]}...")
    print()
```

**Embedding 的工作原理：**

```
文本                          向量（高维数字数组）
"Python 装饰器"    →  [0.12, -0.34, 0.56, ..., 0.78]   768 维
"Python decorator" →  [0.11, -0.33, 0.55, ..., 0.77]   768 维  ← 语义相近，向量也接近！
"今天天气不错"      →  [0.89, 0.23, -0.67, ..., 0.01]   768 维  ← 语义不同，向量差异大

搜索时：把用户问题也转成向量，找最"接近"的文档块

```

**Chroma vs FAISS：**

| 特性     | Chroma                         | FAISS                   |
| :------- | :----------------------------- | :---------------------- |
| 安装     | `pip install langchain-chroma` | `pip install faiss-cpu` |
| 持久化   | ✅ 内置                         | ❌ 需手动 save/load      |
| 过滤     | ✅ 支持 metadata 过滤           | ❌ 不支持                |
| 适用场景 | 中小规模（万级文档）           | 大规模（百万级文档）    |

> 💡 **小项目用 Chroma**（开箱即用，支持持久化和过滤），**大规模生产用 FAISS 或 Milvus**。

#### 5.5 检索链：从"搜到"到"回答"

有了向量数据库，就可以构建完整的 RAG 链——**先检索、再回答**：

```python
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatDeepSeek(model="deepseek-chat")

# ── 创建检索器 ──
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度搜索
    search_kwargs={"k": 3},     # 返回最相关的 3 个文档块
)

# ── RAG Prompt 模板 ──
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识库助手。请根据以下参考文档回答用户问题。
                  如果文档中没有相关信息，请明确告知用户"我没有找到相关信息"，不要编造。
                  参考文档：
                  {context}"""),
    ("human", "{question}"),]
)

# ── 辅助函数：把检索到的文档格式化为字符串 ──
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}"
        for doc in docs
    )

# ── 组装 RAG 链 ──
rag_chain = (
    {
        "context": retriever | format_docs,   # 检索 → 格式化
        "question": RunnablePassthrough(),     # 用户问题直接传递
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ── 使用 ──
answer = rag_chain.invoke("Python 装饰器的 @语法糖是怎么工作的？")
print(answer)
# 根据参考文档，@语法糖实际上是 func = decorator(func) 的简写...
```

**RAG 链的数据流：**

```
"装饰器怎么工作？"
      │
      ├──→ retriever.invoke()
      │      → 向量搜索
      │      → 返回 3 个相关文档块
      │      → format_docs() 拼成字符串
      │      → {"context": "文档内容..."}
      │
      ├──→ RunnablePassthrough()
      │      → {"question": "装饰器怎么工作？"}
      │
      ▼
 合并为 {"context": "...", "question": "..."}
      │
      ▼
 rag_prompt → LLM → StrOutputParser → 最终回答
```

#### 5.6 实战：基于本地文档的智能问答系统

把前面所有步骤串起来，构建一个完整的"本地文档问答系统"：

```python
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════
# 离线阶段：文档 → 向量数据库（只需运行一次）
# ════════════════════════════════════════════
def build_knowledge_base(docs_dir: str, db_dir: str = "./chroma_db"):
    """构建知识库"""
    # 1. 加载文档
    loader = DirectoryLoader(docs_dir, glob="**/*.md")
    docs = loader.load()
    print(f"📄 加载了 {len(docs)} 个文档")
    
    # 2. 分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✂️ 分割成 {len(chunks)} 个块")
    
    # 3. 向量化 + 存储
    embeddings = HuggingFaceEmbeddings(
        model_name="model/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=db_dir)
    print(f"💾 已存入向量数据库: {db_dir}")
    return vectorstore

# ════════════════════════════════════════════
# 在线阶段：问答（每次提问调用）
# ════════════════════════════════════════════
def create_qa_chain(vectorstore):
    """创建问答链"""
    llm = ChatDeepSeek(model="deepseek-chat")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        ("system", """基于以下参考文档回答问题。引用来源时标注文件名。
        如果文档中没有相关信息，回答"我没有找到相关信息"。
        
        参考文档：
        {context}"""
        ),
        ("human", "{question}"),
    ])
    
    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"【{doc.metadata.get('source', '?')}】\n{doc.page_content}"
            for doc in docs
        )
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

# ════════════════════════════════════════════
# 使用
# ════════════════════════════════════════════

# 首次运行：构建知识库
# vectorstore = build_knowledge_base("./my_docs/")

# 后续运行：加载已有知识库
embeddings = HuggingFaceEmbeddings(
        model_name="model/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

qa = create_qa_chain(vectorstore)
print(qa.invoke("本系统中电子书城的核心功能包括哪些？"))

```

**第 5 章核心知识回顾：**

| 步骤   | 组件             | 一句话解释                    |
| :----- | :--------------- | :---------------------------- |
| 加载   | `DocumentLoader` | 把文件/网页变成 Document 对象 |
| 分割   | `TextSplitter`   | 把长文档切成小块              |
| 向量化 | `Embeddings`     | 把文本转成高维数字向量        |
| 存储   | `VectorStore`    | 存储向量，支持相似度搜索      |
| 检索   | `Retriever`      | 根据问题找到最相关的文档块    |
| 回答   | `LLM + Prompt`   | 把文档+问题组合，生成回答     |