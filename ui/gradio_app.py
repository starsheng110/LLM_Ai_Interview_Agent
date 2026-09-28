"""
展示层 - Gradio Web 界面
"""
import gradio as gr
from services.resume_service import analyze_resume_info
from services.interview_service import interview_conversation, clear_session
from services.report_service import interview_report


def build_demo():
    """
    构建 Gradio 界面

    Returns:
        Gradio Blocks 实例
    """
    with gr.Blocks() as demo:
        gr.Markdown("# AI 智能面试平台")
        gr.Markdown("### 功能: 简历解析 | RAG 智能面试 | 自动评估面试报告 | 数据管理")

        with gr.Tabs():
            with gr.Tab('面试官'):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 候选人简历上传")
                        resume_input = gr.File(label='上传简历PDF')
                        analyze_resume_info_btn = gr.Button(
                            value="解析简历信息", variant='primary'
                        )
                        interview_direction = gr.Dropdown(
                            label="面试方向",
                            choices=['软件测试', 'Java工程师', 'AI-Agent工程师'],
                            multiselect=False,
                            interactive=True
                        )
                        resume_output = gr.TextArea(label='简历提取信息', lines=15)
                        with gr.Row():
                            gr.Button(value="重新构建面试题库知识库")
                        with gr.Row():
                            gr.Textbox(label='知识库状态')

                    with gr.Column(scale=2):
                        gr.Markdown("## AI 面试官对话区")
                        with gr.Row():
                            with gr.Column(scale=1):
                                question_count = gr.Number(
                                    label='面试题数量',
                                    value=4,
                                    precision=0,
                                    minimum=0,
                                    maximum=15,
                                    interactive=True
                                )
                            with gr.Column(scale=2):
                                difficulty_level = gr.Dropdown(
                                    choices=["高", "中", "低"],
                                    value="中",
                                    label="请选择难度等级",
                                    multiselect=False,
                                    interactive=True
                                )

                        chatbot = gr.Chatbot(label='面试对话', height=450)
                        user_reply = gr.Textbox(
                            label='你的回答/指令(输入"开始面试"启动，"结束面试"停止)'
                        )
                        reply_btn = gr.Button(value="发送回答", variant='primary')

                with gr.Row():
                    with gr.Column(scale=1):
                        report_btn = gr.Button(value="生成面试评估报告")
                    with gr.Column(scale=1):
                        save_session_btn = gr.Button(value="保存当前会话")
                    with gr.Column(scale=1):
                        clear_session_btn = gr.Button(value="清空全部会话")
                with gr.Row():
                    final_report_output = gr.Markdown(value='面试综合评估报告')

            with gr.Tab('数据分析'):
                gr.Markdown("## 待开发")

        # ============ 事件绑定 ============
        analyze_resume_info_btn.click(
            fn=analyze_resume_info,
            inputs=[resume_input],
            outputs=[resume_output]
        )

        reply_btn.click(
            fn=interview_conversation,
            inputs=[
                resume_output,
                interview_direction,
                question_count,
                difficulty_level,
                user_reply,
                chatbot,
            ],
            outputs=[user_reply, chatbot, question_count]
        )

        report_btn.click(
            fn=interview_report,
            inputs=[resume_output, chatbot, interview_direction, difficulty_level],
            outputs=[final_report_output]
        )

        clear_session_btn.click(
            fn=clear_session,
            inputs=[],
            outputs=[chatbot, user_reply, question_count, final_report_output]
        )

    return demo
