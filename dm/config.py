"""Runtime configuration, read lazily from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
LORE_PATH = ROOT / "world" / "lore.json"

load_dotenv(ROOT / ".env")


def backend_name() -> str:
    return os.getenv("DM_BACKEND") or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "mock")


def model() -> str:
    return os.getenv("DM_MODEL", "claude-sonnet-5-5")


def openai_base_url() -> str:
    return os.getenv("DM_OPENAI_BASE_URL", "http://localhost:8000/v1")


def openai_api_key() -> str:
    return os.getenv("DM_OPENAI_API_KEY", "none")


def adapter_path() -> Path:
    return Path(os.getenv("DM_ADAPTER_PATH", ROOT / "models" / "dm-lora"))


def base_model() -> str:
    return os.getenv("DM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def data_dir() -> Path:
    return Path(os.getenv("DM_DATA_DIR", ROOT / "data"))


def use_dense() -> bool:
    return os.getenv("DM_DENSE", "0") == "1"


def use_rerank() -> bool:
    return os.getenv("DM_RERANK", "0") == "1"


def dataset_path():
    """If set, every turn is logged as a fine-tuning example."""
    p = os.getenv("DM_LOG_DATASET")
    return Path(p) if p else None
