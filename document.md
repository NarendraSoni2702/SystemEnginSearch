original program flow pattern was like that 
main()
├── configuration
├── HTTP client
├── file reading
├── text cleaning
├── chunking
├── hashing
├── cache management
├── embedding API
├── vector search
├── prompt construction
├── LLM API
└── interactive UI

which will be converted into structured framework
Application
│
├── FileReader
│
├── TextChunker
│
├── IndexStore
│
├── EmbeddingService
│       └── OpenRouterClient
│
├── SemanticRetriever
│       └── EmbeddingService
│
└── RAGAssistant
    └── OpenRouterClient



AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_CHAT_DEPLOYMENT
AZURE_OPENAI_EMBEDDING_DEPLOYMENT

For example, in PowerShell:

$env:AZURE_OPENAI_ENDPOINT="https://my-resource.openai.azure.com"
$env:AZURE_OPENAI_API_KEY="your-api-key"
$env:AZURE_OPENAI_CHAT_DEPLOYMENT="my-chat-deployment"
$env:AZURE_OPENAI_EMBEDDING_DEPLOYMENT="my-embedding-deployment"

Question
↓
Embedding
↓
Cosine similarity
↓
Remove weak matches
↓
Take top K
↓
LLM