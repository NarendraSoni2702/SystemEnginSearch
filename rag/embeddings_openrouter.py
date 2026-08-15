import numpy as np

from llm.openrouter import OpenRouterClient
from models import Chunk


class EmbeddingService:

    def __init__(
            self,
            client: OpenRouterClient,
            model: str,
            batch_size: int = 8,
    ):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed_chunks(
            self,
            chunks: list[Chunk],
    ) -> np.ndarray:

        texts = [
            chunk.content
            for chunk in chunks
        ]

        return self._embed_batches(texts)

    def embed_query(
            self,
            question: str,
    ) -> np.ndarray:

        return self.client.embeddings(
            [question],
            self.model,
        )[0]

    def _embed_batches(
            self,
            texts: list[str],
    ) -> np.ndarray:

        embeddings = []

        total = len(texts)

        for start in range(
                0,
                total,
                self.batch_size,
        ):
            batch = texts[
                start:start + self.batch_size
            ]

            end = min(
                start + self.batch_size,
                total,
                )

            print(
                f"\rEmbedding {end}/{total}...",
                end="",
                flush=True,
            )

            result = self.client.embeddings(
                batch,
                self.model,
            )

            embeddings.extend(result)

        print()

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )