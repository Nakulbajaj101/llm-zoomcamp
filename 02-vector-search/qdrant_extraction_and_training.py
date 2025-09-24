import requests
from qdrant_client import models
from utility_functions import QdrantVectorStore
from config import QDRANT_HOST, DOCS_URL, EMBEDDING_DIMENSIONALITY, MODEL_HANDLE, COLLECTION_NAME


def fetch_docs(url: str):
    response = requests.get(url)
    return response.json()


def upsert_documents(client, collection_name: str, documents: list, model_handle: str = MODEL_HANDLE):
    points = []
    id = 0
    for course in documents:
        for doc in course['documents']:
            point = models.PointStruct(
                id=id,
                vector=models.Document(text=str(doc['text']), model=model_handle),
                payload={
                    "text": doc["text"],
                    "section": doc["section"],
                    "course": course['course']
                }
            )
            points.append(point)
            id += 1

    client.upsert(collection_name=collection_name,
                        points=points)
    

if __name__ == "__main__":
    # Fetch documents
    documents_raw = fetch_docs(DOCS_URL)

    # Initialize Qdrant client
    qv = QdrantVectorStore(host=QDRANT_HOST)

    # Create collection
    qv.create_collection(collection_name=COLLECTION_NAME,
                                   embedding_dimensionality=EMBEDDING_DIMENSIONALITY)

    # Upsert documents
    upsert_documents(client=qv.client, collection_name=COLLECTION_NAME, documents=documents_raw, model_handle=MODEL_HANDLE)