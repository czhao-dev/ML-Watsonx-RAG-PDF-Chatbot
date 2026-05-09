"""Gradio interface for the portfolio RAG application."""

import gradio as gr

from app.rag_pipeline import answer_question


def build_interface() -> gr.Interface:
    return gr.Interface(
        fn=answer_question,
        allow_flagging="never",
        inputs=[
            gr.File(
                label="Upload PDF",
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
            ),
            gr.Textbox(
                label="Question",
                lines=2,
                placeholder="Ask a question about the uploaded document...",
            ),
        ],
        outputs=gr.Textbox(label="Answer"),
        title="RAG PDF QA Chatbot",
        description=(
            "Upload a PDF and ask questions answered from retrieved document context."
        ),
    )
