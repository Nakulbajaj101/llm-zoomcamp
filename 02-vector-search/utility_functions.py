from qdrant_client import QdrantClient, models

class QdrantVectorStore:
    def __init__(self, host: str):
        self.client = QdrantClient(url=host)

    def create_collection(self, collection_name: str, embedding_dimensionality: int):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(collection_name=collection_name,
                                        vectors_config=models.VectorParams(
                                            size=embedding_dimensionality,
                                            distance=models.Distance.COSINE
                                        )
                                        )
        else:
            print(f"Collection {collection_name} already exists.")
            