import uuid

import requests
from config import COURSES_URL, QDRANT_HOST, MODEL_HANDLE, EMBEDDING_DIMENSIONALITY, COLLECTION_NAME
from data_ingestion import fetch_data, fetch_course_faqs
from qdrant_client import models
from utility_functions import QdrantVectorStore


def fetch_docs(url: str):
    response = requests.get(url)
    return response.json()


def upsert_documents(client, collection_name: str, model_handle_dense: str, model_handle_sparse: str, documents: list):
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "jina-small": models.Document(
                        text=doc["answer"],
                        model=model_handle_dense
                        ),
                    "bm25": models.Document(
                        text=doc["answer"],
                        model=model_handle_sparse
                        )
                    },
                payload={
                    "text": doc["answer"],
                    "section": doc["section"],
                    "course": doc['course'],
                    "question": doc['question']
                }
            )
        for doc in documents
        ]
    )
    

if __name__ == "__main__":
    # Fetch documents
    courses = fetch_data(COURSES_URL)
    documents = fetch_course_faqs(courses)
    print(f"Fetched {len(documents)} FAQ entries.")

    # Initialize Qdrant client
    qv = QdrantVectorStore(host=QDRANT_HOST)

    # Create collection
    qv.create_collection_hybrid(collection_name=COLLECTION_NAME, embedding_dimensionality=EMBEDDING_DIMENSIONALITY)

    # Upsert documents
    upsert_documents(client=qv.client,
                     collection_name=COLLECTION_NAME,
                     documents=documents,
                     model_handle_sparse="Qdrant/bm25",
                     model_handle_dense=MODEL_HANDLE
                     )