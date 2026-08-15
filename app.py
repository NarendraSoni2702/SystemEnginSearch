from config import Settings

from readers.file_reader import FileReader

from rag.chunker import TextChunker
from rag.embeddings import EmbeddingService
from rag.index import IndexStore
from rag.retriever import SemanticRetriever

from llm.azure_openai import AzureOpenAIClient

from chat.assistant import RAGAssistant


class Application:

    def __init__(self):

        # ====================================================
        # SETTINGS
        # ====================================================

        self.settings = Settings()

        self.settings.validate()

        # ====================================================
        # FILE READER
        # ====================================================

        self.reader = FileReader()

        # ====================================================
        # CHUNKER
        # ====================================================

        self.chunker = TextChunker(
            chunk_size=(
                self.settings.CHUNK_SIZE
            ),

            chunk_overlap=(
                self.settings.CHUNK_OVERLAP
            ),
        )

        # ====================================================
        # AZURE OPENAI
        # ====================================================

        self.azure = AzureOpenAIClient(

            endpoint=(
                self.settings
                .AZURE_OPENAI_ENDPOINT
            ),

            api_key=(
                self.settings
                .AZURE_OPENAI_API_KEY
            ),

            max_retries=(
                self.settings.MAX_RETRIES
            ),

            retry_base_delay=(
                self.settings
                .RETRY_BASE_DELAY
            ),

            timeout=(
                self.settings
                .REQUEST_TIMEOUT
            ),
        )

        # ====================================================
        # EMBEDDING SERVICE
        # ====================================================

        self.embedding_service = (
            EmbeddingService(

                client=self.azure,

                deployment=(
                    self.settings
                    .AZURE_OPENAI_EMBEDDING_DEPLOYMENT
                ),

                batch_size=(
                    self.settings
                    .EMBED_BATCH_SIZE
                ),
            )
        )

        # ====================================================
        # RETRIEVER
        # ====================================================

        self.retriever = (
            SemanticRetriever(
                self.embedding_service
            )
        )

        # ====================================================
        # INDEX STORE
        # ====================================================

        self.index_store = (
            IndexStore(
                self.settings
            )
        )

        # ====================================================
        # RAG ASSISTANT
        # ====================================================

        self.assistant = (
            RAGAssistant(

                client=self.azure,

                deployment=(
                    self.settings
                    .AZURE_OPENAI_CHAT_DEPLOYMENT
                ),

                temperature=(
                    self.settings.TEMPERATURE
                ),

                max_output_tokens=(
                    self.settings
                    .MAX_OUTPUT_TOKENS
                ),
            )
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            documents = (
                self.load_documents()
            )

            if not documents:

                print()
                print(
                    "No supported documents "
                    "were found."
                )

                print(
                    "Put TXT, CSV, PDF, or DOCX "
                    "files inside the data folder."
                )

                return

            chunks, embeddings = (
                self.prepare_index(
                    documents
                )
            )

            self.print_ready_message(
                documents,
                chunks,
            )

            self.chat_loop(
                chunks,
                embeddings,
            )

        finally:

            self.azure.close()

    # ========================================================
    # LOAD DOCUMENTS
    # ========================================================

    def load_documents(self):

        print()
        print(
            "Loading documents..."
        )

        documents = (
            self.reader.read_directory(
                self.settings.DATA_FOLDER
            )
        )

        print()
        print(
            f"Loaded {len(documents)} "
            f"documents."
        )

        return documents

    # ========================================================
    # PREPARE INDEX
    # ========================================================

    def prepare_index(
            self,
            documents,
    ):

        document_hash = (
            self.index_store
            .calculate_hash(
                documents
            )
        )

        # ----------------------------------------------------
        # Try cache
        # ----------------------------------------------------

        cached = (
            self.index_store.load(
                document_hash
            )
        )

        if cached is not None:

            chunks, embeddings = cached

            print()
            print(
                "Existing index found."
            )

            print(
                f"Loaded {len(chunks)} "
                f"cached chunks."
            )

            return chunks, embeddings

        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        print()
        print(
            "Creating document chunks..."
        )

        chunks = (
            self.chunker.create_chunks(
                documents
            )
        )

        if not chunks:

            raise RuntimeError(
                "No document chunks were created."
            )

        print(
            f"Created {len(chunks)} chunks."
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        print()
        print(
            "Creating Azure OpenAI embeddings..."
        )

        embeddings = (
            self.embedding_service
            .embed_chunks(
                chunks
            )
        )

        if len(embeddings) != len(
                chunks
        ):

            raise RuntimeError(
                "Embedding count does not "
                "match chunk count."
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        print()
        print(
            "Saving local index..."
        )

        self.index_store.save(

            chunks,

            embeddings,

            document_hash,
        )

        print(
            "Index saved."
        )

        return chunks, embeddings

    # ========================================================
    # READY
    # ========================================================

    @staticmethod
    def print_ready_message(
            documents,
            chunks,
    ):

        print()
        print(
            "=========================================="
        )

        print(
            "       AZURE OPENAI RAG ASSISTANT"
        )

        print(
            "=========================================="
        )

        print()

        print(
            f"Documents: {len(documents)}"
        )

        print(
            f"Chunks:    {len(chunks)}"
        )

        print()

        print(
            "Ask questions about your documents."
        )

        print(
            "Type 'exit' to quit."
        )

        print()

    # ========================================================
    # CHAT LOOP
    # ========================================================

    def chat_loop(
            self,
            chunks,
            embeddings,
    ):

        while True:

            try:

                question = input(
                    "Ask a question: "
                ).strip()

            except (
                    KeyboardInterrupt,
                    EOFError,
            ):

                print()
                print(
                    "Goodbye!"
                )

                break

            if not question:
                continue

            if (
                    question.lower()
                    == "exit"
            ):

                print(
                    "Goodbye!"
                )

                break

            if self.is_greeting(
                    question
            ):

                print()
                print(
                    "AI: Hello! How can I "
                    "help you with your files?"
                )
                print()

                continue

            self.answer_question(
                question,
                chunks,
                embeddings,
            )

    # ========================================================
    # ANSWER QUESTION
    # ========================================================

    def answer_question(
            self,
            question,
            chunks,
            embeddings,
    ):

        try:

            print()
            print(
                "Searching documents..."
            )

            results = (
                self.retriever.search(

                    question=question,

                    chunks=chunks,

                    embeddings=embeddings,

                    top_k=(
                        self.settings.TOP_K
                    ),

                    min_similarity=(
                        self.settings
                        .MIN_SIMILARITY
                    ),
                )
            )

            if not results:

                print()
                print(
                    "AI:"
                )

                print(
                    "I couldn't find that "
                    "information in the files."
                )

                print()

                return

            self.display_results(
                results
            )

            print()
            print(
                "Asking Azure OpenAI..."
            )

            answer = (
                self.assistant.answer(
                    question,
                    results,
                )
            )

            print()
            print(
                "AI:"
            )

            print(
                answer
            )

            print()

        except Exception as error:

            print()
            print(
                "ERROR:"
            )

            print(
                error
            )

            print()

    # ========================================================
    # GREETING
    # ========================================================

    @staticmethod
    def is_greeting(
            question: str,
    ) -> bool:

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

        return (
                question.lower().strip()
                in greetings
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    @staticmethod
    def display_results(
            results,
    ):

        print()
        print(
            "Retrieved sources:"
        )

        for number, result in enumerate(
                results,
                start=1,
        ):

            print(
                f"  {number}. "
                f"{result.filename} "
                f"({result.score:.3f})"
            )

        print()


if __name__ == "__main__":

    Application().run()