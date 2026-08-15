import time
from pathlib import Path

import streamlit as st

import rag_engine


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Local AI File Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        margin-bottom: 1.5rem;
    }

    .source-box {
        background-color: #f5f7fa;
        border-left: 4px solid #4f8cff;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 8px;
    }

    .status-box {
        padding: 10px;
        border-radius: 8px;
        background-color: #f5f7fa;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "engine" not in st.session_state:

    st.session_state.engine = None


if "engine_documents_hash" not in st.session_state:

    st.session_state.engine_documents_hash = None


if "selected_files" not in st.session_state:

    st.session_state.selected_files = set()


if "force_rebuild" not in st.session_state:

    st.session_state.force_rebuild = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file):
    """Save uploaded Streamlit file into data folder."""

    rag_engine.DATA_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
            rag_engine.DATA_FOLDER
            / Path(uploaded_file.name).name
    )

    with open(
            destination,
            "wb",
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return destination


def delete_document(filename):
    """Delete document from data folder."""

    file_path = (
            rag_engine.DATA_FOLDER
            / filename
    )

    if file_path.exists():

        file_path.unlink()

        return True

    return False


def get_current_documents():
    """Load documents using the RAG engine."""

    return rag_engine.load_documents()


def get_current_document_names():
    """Return filenames currently in data folder."""

    return [
        document["filename"]
        for document in get_current_documents()
    ]


def calculate_current_hash():
    """Calculate current document hash."""

    documents = get_current_documents()

    if not documents:
        return None

    return rag_engine.calculate_documents_hash(
        documents
    )


def invalidate_engine():
    """Clear cached engine from Streamlit session."""

    st.session_state.engine = None

    st.session_state.engine_documents_hash = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 📁 Documents"
    )

    st.caption(
        "Manage documents used by the local RAG assistant."
    )

    # --------------------------------------------------------
    # OLLAMA STATUS
    # --------------------------------------------------------

    if rag_engine.check_ollama():

        st.success(
            "Ollama is running"
        )

    else:

        st.error(
            "Ollama is not running"
        )

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    models = rag_engine.get_ollama_models()

    if rag_engine.LLM_MODEL in models:

        st.caption(
            f"🤖 LLM: `{rag_engine.LLM_MODEL}`"
        )

    else:

        st.warning(
            f"Missing LLM: "
            f"{rag_engine.LLM_MODEL}"
        )

    if rag_engine.EMBEDDING_MODEL in models:

        st.caption(
            f"🔎 Embeddings: "
            f"`{rag_engine.EMBEDDING_MODEL}`"
        )

    else:

        st.warning(
            "Missing embedding model: "
            f"{rag_engine.EMBEDDING_MODEL}"
        )

    st.divider()

    # ========================================================
    # UPLOAD DOCUMENTS
    # ========================================================

    st.markdown(
        "### ➕ Add Documents"
    )

    uploaded_files = st.file_uploader(
        "Upload files",
        type=[
            "txt",
            "csv",
            "pdf",
            "docx",
        ],
        accept_multiple_files=True,
    )

    if st.button(
            "📥 Add uploaded documents",
            use_container_width=True,
    ):

        if not uploaded_files:

            st.warning(
                "Please select at least one file."
            )

        else:

            added = 0

            for uploaded_file in uploaded_files:

                save_uploaded_file(
                    uploaded_file
                )

                added += 1

            invalidate_engine()

            st.success(
                f"Added {added} document(s)."
            )

            time.sleep(0.5)

            st.rerun()

    st.divider()

    # ========================================================
    # DOCUMENT LIST
    # ========================================================

    documents = get_current_documents()

    current_files = [
        document["filename"]
        for document in documents
    ]

    st.markdown(
        f"### Documents ({len(documents)})"
    )

    # Remove deleted files from selection.
    st.session_state.selected_files = (
            st.session_state.selected_files
            & set(current_files)
    )

    if documents:

        for document in documents:

            filename = document["filename"]

            selected = st.checkbox(
                filename,
                value=(
                        filename
                        in st.session_state.selected_files
                ),
                key=f"file_{filename}",
            )

            if selected:

                st.session_state.selected_files.add(
                    filename
                )

            else:

                st.session_state.selected_files.discard(
                    filename
                )

    else:

        st.info(
            "No documents found."
        )

    st.divider()

    # ========================================================
    # SELECT ALL / NONE
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
                "☑ All",
                use_container_width=True,
        ):

            st.session_state.selected_files = set(
                current_files
            )

            st.rerun()

    with col2:

        if st.button(
                "☐ None",
                use_container_width=True,
        ):

            st.session_state.selected_files = set()

            st.rerun()

    st.divider()

    # ========================================================
    # DELETE SELECTED
    # ========================================================

    if st.button(
            "🗑 Delete selected",
            use_container_width=True,
    ):

        selected = list(
            st.session_state.selected_files
        )

        if not selected:

            st.warning(
                "No documents selected."
            )

        else:

            deleted = 0

            for filename in selected:

                if delete_document(
                        filename
                ):

                    deleted += 1

            st.session_state.selected_files = set()

            invalidate_engine()

            st.success(
                f"Deleted {deleted} document(s)."
            )

            time.sleep(0.5)

            st.rerun()

    # ========================================================
    # INDEX
    # ========================================================

    st.markdown(
        "### 🔎 Index"
    )

    if st.button(
            "🔄 Rebuild index",
            use_container_width=True,
    ):

        invalidate_engine()

        st.session_state.force_rebuild = True

        st.rerun()

    if st.button(
            "🧹 Clear chat",
            use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # ========================================================
    # SETTINGS
    # ========================================================

    with st.expander(
            "⚙️ Settings"
    ):

        st.write(
            f"**LLM:** `{rag_engine.LLM_MODEL}`"
        )

        st.write(
            f"**Embedding:** "
            f"`{rag_engine.EMBEDDING_MODEL}`"
        )

        st.write(
            f"**Top K:** `{rag_engine.TOP_K}`"
        )

        st.write(
            f"**Chunk size:** "
            f"`{rag_engine.CHUNK_SIZE}`"
        )

        st.write(
            f"**Chunk overlap:** "
            f"`{rag_engine.CHUNK_OVERLAP}`"
        )

        st.write(
            f"**Context:** "
            f"`{rag_engine.NUM_CTX}`"
        )

        st.write(
            f"**Max output:** "
            f"`{rag_engine.NUM_PREDICT}`"
        )

        st.write(
            f"**CPU threads:** "
            f"`{rag_engine.OLLAMA_NUM_THREADS}`"
        )

        st.write(
            f"**Keep alive:** "
            f"`{rag_engine.OLLAMA_KEEP_ALIVE}`"
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🤖 Local AI File Assistant'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your local documents using Ollama.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DOCUMENT STATUS
# ============================================================

documents = get_current_documents()

if not documents:

    st.info(
        "👈 Add documents using the sidebar to get started."
    )

    st.stop()


# ============================================================
# PREPARE ENGINE
# ============================================================

documents_hash = rag_engine.calculate_documents_hash(
    documents
)

force_rebuild = st.session_state.pop(
    "force_rebuild",
    False,
)

needs_initialization = (
        st.session_state.engine is None
        or
        st.session_state.engine_documents_hash
        != documents_hash
        or
        force_rebuild
)


if needs_initialization:

    progress_placeholder = st.empty()

    def progress_callback(message):

        progress_placeholder.info(
            f"🔎 {message}"
        )

    try:

        engine = rag_engine.initialize(
            progress_callback=progress_callback,
            force_rebuild=force_rebuild,
        )

        st.session_state.engine = engine

        st.session_state.engine_documents_hash = (
            documents_hash
        )

        progress_placeholder.empty()

    except Exception as error:

        progress_placeholder.empty()

        st.error(
            f"Could not prepare RAG engine: {error}"
        )

        st.stop()


# ============================================================
# ENGINE
# ============================================================

engine = st.session_state.engine


# ============================================================
# STATUS CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Documents",
        len(engine["documents"]),
    )

with col2:

    st.metric(
        "Chunks",
        len(engine["chunks"]),
    )

with col3:

    selected_count = len(
        st.session_state.selected_files
    )

    st.metric(
        "Selected",
        selected_count,
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
            message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
                message["role"] == "assistant"
                and message.get("sources")
        ):

            st.markdown(
                '<div class="source-box">'
                "📄 Sources: "
                + ", ".join(
                    message["sources"]
                )
                + "</div>",
                unsafe_allow_html=True,
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Determine document scope
    # --------------------------------------------------------

    selected_files = (
        st.session_state.selected_files
    )

    if not selected_files:

        selected_files_for_search = None

    else:

        selected_files_for_search = (
            selected_files
        )

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        search_placeholder = st.empty()

        search_placeholder.info(
            "🔎 Searching documents..."
        )

        try:

            result = rag_engine.ask_question(
                question,
                engine,
                selected_files=selected_files_for_search,
            )

        except Exception as error:

            search_placeholder.error(
                f"Error: {error}"
            )

            st.stop()

        search_placeholder.empty()

        # ----------------------------------------------------
        # Search timing
        # ----------------------------------------------------

        st.caption(
            f"Semantic search: "
            f"{result.get('search_time', 0):.2f} seconds"
        )

        # ----------------------------------------------------
        # Retrieved sources
        # ----------------------------------------------------

        retrieved_sources = result.get(
            "sources",
            []
        )

        source_names = []

        for source in retrieved_sources:

            filename = source["filename"]

            if filename not in source_names:

                source_names.append(
                    filename
                )

        if retrieved_sources:

            with st.expander(
                    "📚 Retrieved sources"
            ):

                for number, source in enumerate(
                        retrieved_sources,
                        start=1,
                ):

                    st.write(
                        f"**{number}. "
                        f"{source['filename']}** "
                        f"— similarity "
                        f"{source['score']:.3f}"
                    )

        # ----------------------------------------------------
        # AI answer
        # ----------------------------------------------------

        answer = result.get(
            "answer",
            "",
        )

        if not answer:

            answer = (
                "I couldn't find that information "
                "in the files."
            )

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # AI performance
        # ----------------------------------------------------

        ai_time = result.get(
            "ai_time"
        )

        if ai_time is not None:

            st.caption(
                f"AI generation: "
                f"{ai_time:.1f} seconds"
            )

        prompt_tokens = result.get(
            "prompt_tokens"
        )

        generated_tokens = result.get(
            "generated_tokens"
        )

        prompt_time = result.get(
            "prompt_time"
        )

        generation_time = result.get(
            "generation_time"
        )

        if prompt_tokens is not None:

            st.caption(
                f"Prompt tokens: {prompt_tokens} "
                f"| Generated tokens: "
                f"{generated_tokens}"
            )

        if (
                prompt_time is not None
                and generation_time is not None
        ):

            st.caption(
                f"Prompt processing: "
                f"{prompt_time:.2f}s "
                f"| Token generation: "
                f"{generation_time:.2f}s"
            )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if source_names:

            st.markdown(
                '<div class="source-box">'
                "📄 Sources: "
                + ", ".join(source_names)
                + "</div>",
                unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # Save assistant message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_names,
            }
        )