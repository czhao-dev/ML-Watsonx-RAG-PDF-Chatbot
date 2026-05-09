"""Application settings loaded from environment variables."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    watsonx_url: str
    watsonx_project_id: str
    watsonx_api_key: str | None
    llm_model_id: str
    embedding_model_id: str
    max_new_tokens: int
    temperature: float
    chunk_size: int
    chunk_overlap: int
    server_name: str
    server_port: int


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def get_settings() -> AppSettings:
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "WATSONX_PROJECT_ID is required. Copy .env.example to .env and set your watsonx project ID."
        )

    return AppSettings(
        watsonx_url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        watsonx_project_id=project_id,
        watsonx_api_key=os.getenv("WATSONX_APIKEY") or None,
        llm_model_id=os.getenv("WATSONX_LLM_MODEL_ID", "ibm/granite-3-2-8b-instruct"),
        embedding_model_id=os.getenv(
            "WATSONX_EMBEDDING_MODEL_ID", "ibm/slate-125m-english-rtrvr-v2"
        ),
        max_new_tokens=_get_int("MAX_NEW_TOKENS", 256),
        temperature=_get_float("TEMPERATURE", 0.5),
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 20),
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=_get_int("GRADIO_SERVER_PORT", 7860),
    )
