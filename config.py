import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:

    # ========================================================
    # FILES
    # ========================================================

    DATA_FOLDER = Path("data")
    INDEX_FOLDER = Path("index")

    # ========================================================
    # AZURE OPENAI
    # ========================================================

    AZURE_OPENAI_ENDPOINT = os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "",
    ).strip()

    AZURE_OPENAI_API_KEY = os.getenv(
        "AZURE_OPENAI_API_KEY",
        "",
    ).strip()

    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv(
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "",
    ).strip()

    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "",
    ).strip()

    # ========================================================
    # RAG
    # ========================================================

    TOP_K = 5

    # Ignore very weak semantic matches.
    MIN_SIMILARITY = 0.35

    # ========================================================
    # CHUNKING
    # ========================================================

    CHUNK_SIZE = 1200

    CHUNK_OVERLAP = 200

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    EMBED_BATCH_SIZE = 16

    # ========================================================
    # LLM
    # ========================================================

    MAX_OUTPUT_TOKENS = 300

    TEMPERATURE = 0

    # ========================================================
    # NETWORK
    # ========================================================

    MAX_RETRIES = 4

    RETRY_BASE_DELAY = 2.0

    REQUEST_TIMEOUT = 300.0

    # ========================================================
    # CACHE FILES
    # ========================================================

    @property
    def chunks_file(self):
        return self.INDEX_FOLDER / "chunks.json"

    @property
    def embeddings_file(self):
        return self.INDEX_FOLDER / "embeddings.npy"

    @property
    def cache_info_file(self):
        return self.INDEX_FOLDER / "cache_info.json"

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self):

        missing = []

        if not self.AZURE_OPENAI_ENDPOINT:
            missing.append(
                "AZURE_OPENAI_ENDPOINT"
            )

        if not self.AZURE_OPENAI_API_KEY:
            missing.append(
                "AZURE_OPENAI_API_KEY"
            )

        if not self.AZURE_OPENAI_CHAT_DEPLOYMENT:
            missing.append(
                "AZURE_OPENAI_CHAT_DEPLOYMENT"
            )

        if not self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
            missing.append(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            )

        if missing:

            raise RuntimeError(
                "Missing environment variables:\n\n"
                + "\n".join(
                    f"  - {item}"
                    for item in missing
                )
            )

        if not (
                self.AZURE_OPENAI_ENDPOINT.startswith(
                    "https://"
                )
        ):

            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT must start "
                "with https://"
            )