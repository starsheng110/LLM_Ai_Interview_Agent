"""调试 Chroma 加载"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('../.env')

db_dir = "../chroma_db"
print(f"db_dir 绝对路径: {os.path.abspath(db_dir)}")
print(f"目录存在: {os.path.exists(db_dir)}")

print("\n[1/3] 加载 embedding 模型...")
t0 = time.time()
from rag.embeddings import get_embeddings
embeddings = get_embeddings()
print(f"  完成，耗时 {time.time()-t0:.1f}s")

print("\n[2/3] 加载 Chroma...")
t0 = time.time()
from langchain_chroma import Chroma
vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
print(f"  完成，耗时 {time.time()-t0:.1f}s")

print("\n[3/3] 测试检索...")
t0 = time.time()
retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"topic": "Java面试"}, "k": 2}
)
docs = retriever.invoke("")
print(f"  完成，耗时 {time.time()-t0:.1f}s")
print(f"  检索到 {len(docs)} 条结果")
if docs:
    print(f"  第一条前100字: {docs[0].page_content[:100]}...")

print("\n🎉 Chroma 加载和检索正常！")
