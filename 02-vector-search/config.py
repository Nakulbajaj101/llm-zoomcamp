import os

QDRANT_HOST = os.getenv("QDRANT_CLIENT", "http://localhost:6333")
DOCS_URL = os.getenv("DOCS_URL", 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json')
EMBEDDING_DIMENSIONALITY = 512
MODEL_HANDLE = "jinaai/jina-embeddings-v2-small-en"
COLLECTION_NAME = "zoomcamo-rag"