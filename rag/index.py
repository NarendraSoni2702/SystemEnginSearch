import hashlib
import json

import numpy as np

from config import Settings

from models import Chunk, Document


class IndexStore:

    def __init__(
            self,
            settings: Settings,
    ):

        self.settings = settings

    # ========================================================
    # HASH
    # ========================================================

    def calculate_hash(
            self,
            documents: list[Document],
    ) -> str:

        hasher = hashlib.sha256()

        configuration = [
            str(
                self.settings.CHUNK_SIZE
            ),
            str(
                self.settings.CHUNK_OVERLAP
            ),
            self.settings
            .AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        ]

        for value in configuration:

            hasher.update(
                value.encode(
                    "utf-8"
                )
            )

        for document in documents:

            hasher.update(
                document.filename.encode(
                    "utf-8",
                    errors="ignore",
                )
            )

            hasher.update(
                document.content.encode(
                    "utf-8",
                    errors="ignore",
                )
            )

        return hasher.hexdigest()

    # ========================================================
    # SAVE
    # ========================================================

    def save(
            self,
            chunks: list[Chunk],
            embeddings: np.ndarray,
            document_hash: str,
    ):

        self.settings.INDEX_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Chunks
        # ----------------------------------------------------

        serialized_chunks = [

            {
                "chunk_id":
                    chunk.chunk_id,

                "filename":
                    chunk.filename,

                "content":
                    chunk.content,

                "start":
                    chunk.start,

                "end":
                    chunk.end,
            }

            for chunk in chunks
        ]

        with open(
                self.settings.chunks_file,
                "w",
                encoding="utf-8",
        ) as file:

            json.dump(
                serialized_chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        np.save(
            self.settings.embeddings_file,
            embeddings,
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {

            "documents_hash":
                document_hash,

            "embedding_deployment":
                self.settings
                .AZURE_OPENAI_EMBEDDING_DEPLOYMENT,

            "chunk_size":
                self.settings.CHUNK_SIZE,

            "chunk_overlap":
                self.settings.CHUNK_OVERLAP,

            "number_of_chunks":
                len(chunks),

            "embedding_dimensions":
                (
                    int(
                        embeddings.shape[1]
                    )
                    if embeddings.ndim == 2
                    else 0
                ),
        }

        with open(
                self.settings.cache_info_file,
                "w",
                encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
            )

    # ========================================================
    # LOAD
    # ========================================================

    def load(
            self,
            document_hash: str,
    ):

        required_files = [

            self.settings.chunks_file,

            self.settings.embeddings_file,

            self.settings.cache_info_file,
        ]

        if not all(
                path.exists()
                for path in required_files
        ):

            return None

        try:

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            with open(
                    self.settings.cache_info_file,
                    "r",
                    encoding="utf-8",
            ) as file:

                metadata = json.load(file)

            if (
                    metadata.get(
                        "documents_hash"
                    )
                    != document_hash
            ):

                return None

            if (
                    metadata.get(
                        "embedding_deployment"
                    )
                    != self.settings
                    .AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            ):

                return None

            if (
                    metadata.get(
                        "chunk_size"
                    )
                    != self.settings.CHUNK_SIZE
            ):

                return None

            if (
                    metadata.get(
                        "chunk_overlap"
                    )
                    != self.settings.CHUNK_OVERLAP
            ):

                return None

            # ------------------------------------------------
            # Chunks
            # ------------------------------------------------

            with open(
                    self.settings.chunks_file,
                    "r",
                    encoding="utf-8",
            ) as file:

                raw_chunks = json.load(
                    file
                )

            chunks = [

                Chunk(
                    chunk_id=item[
                        "chunk_id"
                    ],

                    filename=item[
                        "filename"
                    ],

                    content=item[
                        "content"
                    ],

                    start=item[
                        "start"
                    ],

                    end=item[
                        "end"
                    ],
                )

                for item in raw_chunks
            ]

            # ------------------------------------------------
            # Embeddings
            # ------------------------------------------------

            embeddings = np.load(
                self.settings.embeddings_file
            )

            if embeddings.ndim != 2:
                return None

            if len(chunks) != len(
                    embeddings
            ):
                return None

            if not chunks:
                return None

            return chunks, embeddings

        except Exception as error:

            print(
                f"Could not load index: "
                f"{error}"
            )

            return None