from config_openrouter import Settings
from readers.file_reader import FileReader
from rag.chunker import TextChunker
from rag.embeddings_openrouter import EmbeddingService
from rag.index import IndexStore
from rag.retriever import SemanticRetriever
from llm.openrouter import OpenRouterClient
from chat.assistant import RAGAssistant


class Application:

    def __init__(self):
        self.settings = Settings()

        self.reader = FileReader()

        self.chunker = TextChunker(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )

        self.client = OpenRouterClient(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.OPENROUTER_URL,
            connect_timeout=(
                self.settings.HTTP_CONNECT_TIMEOUT
            ),
            read_timeout=(
                self.settings.HTTP_READ_TIMEOUT
            ),
            write_timeout=(
                self.settings.HTTP_WRITE_TIMEOUT
            ),
            pool_timeout=(
                self.settings.HTTP_POOL_TIMEOUT
            ),
        )

        self.embedding_service = EmbeddingService(
            client=self.client,
            model=self.settings.EMBEDDING_MODEL,
            batch_size=self.settings.EMBED_BATCH_SIZE,
        )

        self.retriever = SemanticRetriever(
            self.embedding_service
        )

        self.index_store = IndexStore(
            self.settings
        )

        self.assistant = RAGAssistant(
            client=self.client,
            model=self.settings.LLM_MODEL,
            temperature=self.settings.TEMPERATURE,
            max_tokens=self.settings.NUM_PREDICT,
        )

    def run(self):

        try:
            documents = self.load_documents()

            if not documents:
                print(
                    "No supported documents found."
                )
                return

            chunks, embeddings = (
                self.prepare_index(documents)
            )

            self.chat_loop(
                chunks,
                embeddings,
            )

        finally:
            self.client.close()

    def load_documents(self):

        print("Loading files...")

        documents = self.reader.read_directory(
            self.settings.DATA_FOLDER
        )

        print(
            f"Loaded {len(documents)} files."
        )

        return documents

    def prepare_index(self, documents):

        document_hash = (
            self.index_store.calculate_hash(
                documents
            )
        )

        cached = self.index_store.load(
            document_hash
        )

        if cached is not None:
            print(
                "Existing document index found."
            )

            chunks, embeddings = cached

            print(
                f"Loaded {len(chunks)} cached chunks."
            )

            return chunks, embeddings

        print(
            "Splitting documents into chunks..."
        )

        chunks = self.chunker.create_chunks(
            documents
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        print(
            "Creating online embeddings..."
        )

        embeddings = (
            self.embedding_service
            .embed_chunks(chunks)
        )

        self.index_store.save(
            chunks,
            embeddings,
            document_hash,
        )

        print(
            "Document index saved."
        )

        return chunks, embeddings

    def chat_loop(
            self,
            chunks,
            embeddings,
    ):

        print()
        print("Document index ready.")
        print("Type 'exit' to quit.")
        print()

        while True:

            try:
                question = input(
                    "Ask a question: "
                ).strip()

            except (
                    KeyboardInterrupt,
                    EOFError,
            ):
                print("\nGoodbye!")
                break

            if not question:
                continue

            if question.lower() == "exit":
                print("Goodbye!")
                break

            if self.is_greeting(question):
                print(
                    "\nAI: Hello! How can I help "
                    "you with your files?\n"
                )
                continue

            try:
                results = self.retriever.search(
                    question=question,
                    chunks=chunks,
                    embeddings=embeddings,
                    top_k=self.settings.TOP_K,
                )

                if not results:
                    print(
                        "\nI couldn't find relevant "
                        "information in your files.\n"
                    )
                    continue

                self.display_results(results)

                answer = self.assistant.answer(
                    question,
                    results,
                )

                print("\nAI:")
                print(answer)
                print()

            except Exception as exc:
                print(
                    f"\nERROR: {exc}\n"
                )

    @staticmethod
    def is_greeting(question: str) -> bool:

        greetings = {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "bye",
            "good morning",
            "good afternoon",
            "good evening",
        }

        return question.lower().strip() in greetings

    @staticmethod
    def display_results(results):

        print()
        print("Top semantic search results:")
        print()

        for number, result in enumerate(
                results,
                start=1,
        ):
            print(
                f"{number}. "
                f"{result.filename} "
                f"(similarity: "
                f"{result.score:.3f})"
            )


if __name__ == "__main__":
    Application().run()