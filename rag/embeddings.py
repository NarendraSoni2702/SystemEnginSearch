import numpy as np

from llm.azure_openai import AzureOpenAIClient

from models import Chunk


class EmbeddingService:

    def __init__(
            self,
            client: AzureOpenAIClient,
            deployment: str,
            batch_size: int,
    ):

        self.client = client

        self.deployment = deployment

        self.batch_size = batch_size

    # ========================================================
    # DOCUMENT EMBEDDINGS
    # ========================================================

    def embed_chunks(
            self,
            chunks: list[Chunk],
    ) -> np.ndarray:

        texts = [
            chunk.content
            for chunk in chunks
        ]

        embeddings = []

        total = len(texts)

        for start in range(
                0,
                total,
                self.batch_size,
        ):

            batch = texts[
                start:
                start + self.batch_size
            ]

            end = min(
                start + self.batch_size,
                total,
                )

            print(
                f"\rEmbedding "
                f"{end}/{total}...",
                end="",
                flush=True,
            )

            batch_embeddings = (
                self.client.embeddings(
                    batch,
                    self.deployment,
                )
            )

            embeddings.extend(
                batch_embeddings
            )

        print()

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
            self,
            question: str,
    ) -> np.ndarray:

        result = (
            self.client.embeddings(
                [question],
                self.deployment,
            )
        )

        return result[0]