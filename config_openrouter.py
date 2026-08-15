import os
from pathlib import Path


class Settings:
    DATA_FOLDER = Path("data")
    INDEX_FOLDER = Path("index")

    OPENROUTER_URL = "https://openrouter.ai/api/v1"

    LLM_MODEL = "openrouter/free"

    EMBEDDING_MODEL = (
        "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    )

    TOP_K = 3

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150

    EMBED_BATCH_SIZE = 8

    NUM_PREDICT = 150
    TEMPERATURE = 0

    HTTP_CONNECT_TIMEOUT = 20.0
    HTTP_READ_TIMEOUT = 300.0
    HTTP_WRITE_TIMEOUT = 300.0
    HTTP_POOL_TIMEOUT = 20.0

    @property
    def openrouter_api_key(self):
        return os.getenv("OPENROUTER_API_KEY")

    @property
    def chunks_file(self):
        return self.INDEX_FOLDER / "chunks.json"

    @property
    def embeddings_file(self):
        return self.INDEX_FOLDER / "embeddings.npy"

    @property
    def cache_info_file(self):
        return self.INDEX_FOLDER / "cache_info.json"