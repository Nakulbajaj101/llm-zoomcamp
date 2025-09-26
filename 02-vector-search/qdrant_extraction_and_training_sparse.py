import uuid

import requests
from config import DOCS_URL, QDRANT_HOST
from qdrant_client import models
from utility_functions import QdrantVectorStore


def fetch_docs(url: str):
    response = requests.get(url)
    return response.json()


def upsert_documents(client, collection_name: str, model_handle: str, documents: list):
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "bm25": models.Document(
                        text=doc["text"],
                        model=model_handle
                        )
                    },
                payload={
                    "text": doc["text"],
                    "section": doc["section"],
                    "course": course['course']
                }
            )
        for course in documents for doc in course["documents"]
        ]
    )
    

if __name__ == "__main__":
    # Fetch documents
    documents_raw = fetch_docs(DOCS_URL)

    # Initialize Qdrant client
    qv = QdrantVectorStore(host=QDRANT_HOST)

    # Create collection
    COLLECTION_NAME = "zoomcamo-rag-sparse"
    qv.create_collection_sparse(collection_name=COLLECTION_NAME)

    # Upsert documents
    upsert_documents(client=qv.client, collection_name=COLLECTION_NAME, documents=documents_raw, model_handle="Qdrant/bm25")