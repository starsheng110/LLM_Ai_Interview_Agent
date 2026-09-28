"""
AI 智能面试平台 - 入口文件

启动顺序：
1. 加载环境变量
2. 初始化向量知识库（存在则加载，不存在则构建）
3. 注入向量库到工作流节点
4. 构建 Gradio 界面并启动
"""
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# 加载环境变量（.env 中的 API Key 等）
load_dotenv()

from rag.knowledge_base import init_vectorstore
from workflow.nodes import set_vectorstore
from ui.gradio_app import build_demo


def main():
    # ========== 1. 初始化向量知识库 ==========
    db_dir = "chroma_db"
    docs_dir = [
        'files/ai-agent-interview-guide-zh.pdf',
        'files/Java面试宝典.docx',
        'files/软件测试面试宝典.pdf',
    ]

    vectorstore = init_vectorstore(db_dir=db_dir, docs_dir=docs_dir)

    # ========== 2. 注入向量库到工作流节点 ==========
    set_vectorstore(vectorstore)

    # ========== 3. 构建并启动 Gradio 界面 ==========
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
