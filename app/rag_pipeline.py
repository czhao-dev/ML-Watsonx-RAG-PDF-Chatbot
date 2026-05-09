"""RAG pipeline for answering questions from uploaded PDF documents."""

from pathlib import Path

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM

from app.config import AppSettings, get_settings


def _watsonx_kwargs(settings: AppSettings) -> dict[str, str]:
    kwargs = {
        "url": settings.watsonx_url,
        "project_id": settings.watsonx_project_id,
    }
    if settings.watsonx_api_key:
        kwargs["apikey"] = settings.watsonx_api_key
    return kwargs


def build_llm(settings: AppSettings | None = None) -> WatsonxLLM:
    settings = settings or get_settings()
    params = {
        GenParams.MAX_NEW_TOKENS: settings.max_new_tokens,
        GenParams.TEMPERATURE: settings.temperature,
    }
    return WatsonxLLM(
        model_id=settings.llm_model_id,
        params=params,
        **_watsonx_kwargs(settings),
    )


def build_embedding_model(settings: AppSettings | None = None) -> WatsonxEmbeddings:
    settings = settings or get_settings()
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    return WatsonxEmbeddings(
        model_id=settings.embedding_model_id,
        params=embed_params,
        **_watsonx_kwargs(settings),
    )


def load_pdf(pdf_path: str | Path):
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents, settings: AppSettings | None = None):
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_retriever(pdf_path: str | Path, settings: AppSettings | None = None):
    settings = settings or get_settings()
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents, settings)
    vector_store = Chroma.from_documents(chunks, build_embedding_model(settings))
    return vector_store.as_retriever()


def answer_question(pdf_path: str | Path, query: str, settings: AppSettings | None = None) -> str:
    if not pdf_path:
        return "Please upload a PDF file."
    if not query or not query.strip():
        return "Please enter a question."

    settings = settings or get_settings()
    qa_chain = RetrievalQA.from_chain_type(
        llm=build_llm(settings),
        chain_type="stuff",
        retriever=build_retriever(pdf_path, settings),
        return_source_documents=False,
    )
    response = qa_chain.invoke({"query": query.strip()})
    return response["result"]
