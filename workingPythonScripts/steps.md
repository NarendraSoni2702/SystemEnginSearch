Step 1 → Python reads folder
Step 2 → Read PDF/TXT/CSV/DOCX
Step 3 → Convert files into text
Step 4 → Search relevant information
Step 5 → Connect an AI model
Step 6 → Ask questions about your files


PDF / DOCX / CSV / TXT
↓
Local chunking
↓
Local embeddings
↓
Semantic search
↓
Top 2 relevant chunks
↓
OpenRouter FREE LLM
↓
Answer


current system become after using online
YOUR COMPUTER
┌─────────────────────────────┐
│ PDF / DOCX / CSV / TXT      │
│             ↓               │
│        Text extraction      │
│             ↓               │
│          Chunking           │
└─────────────┬───────────────┘
                │
                ↓
    ONLINE EMBEDDING API
                │
                ↓
        NumPy vector search
                │
            Top K chunks
                │
                ↓
            ONLINE LLM
                │
                ↓
                Answer