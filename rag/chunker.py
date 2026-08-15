import re

from models import Chunk, Document


class TextChunker:

    def __init__(
            self,
            chunk_size: int,
            chunk_overlap: int,
    ):

        if chunk_size <= 0:

            raise ValueError(
                "chunk_size must be > 0"
            )

        if (
                chunk_overlap < 0
                or chunk_overlap >= chunk_size
        ):

            raise ValueError(
                "chunk_overlap must be >= 0 "
                "and smaller than chunk_size"
            )

        self.chunk_size = chunk_size

        self.chunk_overlap = (
            chunk_overlap
        )

    # ========================================================
    # PUBLIC
    # ========================================================

    def create_chunks(
            self,
            documents: list[Document],
    ) -> list[Chunk]:

        chunks = []

        chunk_id = 0

        for document in documents:

            document_chunks = (
                self._chunk_document(
                    document,
                    chunk_id,
                )
            )

            chunks.extend(
                document_chunks
            )

            chunk_id += len(
                document_chunks
            )

        return chunks

    # ========================================================
    # DOCUMENT
    # ========================================================

    def _chunk_document(
            self,
            document: Document,
            starting_id: int,
    ) -> list[Chunk]:

        text = document.content

        chunks = []

        start = 0

        chunk_id = starting_id

        length = len(text)

        while start < length:

            target_end = min(
                start + self.chunk_size,
                length,
                )

            end = self._find_boundary(
                text,
                start,
                target_end,
            )

            chunk_text = (
                text[start:end].strip()
            )

            if chunk_text:

                actual_start = (
                    start
                )

                actual_end = (
                    end
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        filename=(
                            document.filename
                        ),
                        content=chunk_text,
                        start=actual_start,
                        end=actual_end,
                    )
                )

                chunk_id += 1

            if end >= length:
                break

            next_start = max(
                end - self.chunk_overlap,
                start + 1,
                )

            start = next_start

        return chunks

    # ========================================================
    # BOUNDARY
    # ========================================================

    @staticmethod
    def _find_boundary(
            text: str,
            start: int,
            target_end: int,
    ) -> int:

        if target_end >= len(text):
            return len(text)

        search_start = max(
            start,
            target_end - 300,
            )

        region = text[
            search_start:target_end
        ]

        # Prefer paragraph boundary.
        positions = [
            match.start()
            for match in re.finditer(
                r"\n\n",
                region,
            )
        ]

        if positions:

            return (
                    search_start
                    + positions[-1]
                    + 2
            )

        # Then sentence boundary.
        positions = [
            match.start()
            for match in re.finditer(
                r"[.!?]\s",
                region,
            )
        ]

        if positions:

            return (
                    search_start
                    + positions[-1]
                    + 2
            )

        # Then whitespace.
        whitespace = region.rfind(
            " "
        )

        if whitespace > 0:

            return (
                    search_start
                    + whitespace
                    + 1
            )

        return target_end