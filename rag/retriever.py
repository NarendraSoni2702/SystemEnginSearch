import numpy as np

from models import Chunk, SearchResult

from rag.embeddings import EmbeddingService


class SemanticRetriever:

    def __init__(
            self,
            embedding_service: EmbeddingService,
    ):

        self.embedding_service = (
            embedding_service
        )

    def search(
            self,
            question: str,
            chunks: list[Chunk],
            embeddings: np.ndarray,
            top_k: int,
            min_similarity: float,
    ) -> list[SearchResult]:

        if not chunks:
            return []

        if len(embeddings) == 0:
            return []

        query_embedding = (
            self.embedding_service
            .embed_query(question)
        )

        query_norm = np.linalg.norm(
            query_embedding
        )

        if query_norm == 0:
            return []

        normalized_query = (
                query_embedding
                / query_norm
        )

        document_norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        document_norms[
            document_norms == 0
            ] = 1.0

        normalized_embeddings = (
                embeddings
                / document_norms
        )

        scores = np.dot(
            normalized_embeddings,
            normalized_query,
        )

        # ----------------------------------------------------
        # Apply similarity threshold first.
        # ----------------------------------------------------

        valid_indices = np.where(
            scores >= min_similarity
        )[0]

        if len(valid_indices) == 0:
            return []

        # ----------------------------------------------------
        # Sort only valid matches.
        # ----------------------------------------------------

        sorted_indices = valid_indices[
            np.argsort(
                scores[valid_indices]
            )[::-1]
        ]

        selected_indices = (
            sorted_indices[:top_k]
        )

        return [
            SearchResult(
                chunk_id=(
                    chunks[index].chunk_id
                ),
                filename=(
                    chunks[index].filename
                ),
                content=(
                    chunks[index].content
                ),
                score=float(
                    scores[index]
                ),
            )
            for index in selected_indices
        ]