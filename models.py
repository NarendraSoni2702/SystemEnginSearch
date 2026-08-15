from dataclasses import dataclass


@dataclass(frozen=True)
class Document:

    filename: str

    content: str


@dataclass(frozen=True)
class Chunk:

    chunk_id: int

    filename: str

    content: str

    start: int

    end: int


@dataclass(frozen=True)
class SearchResult:

    chunk_id: int

    filename: str

    content: str

    score: float