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

    def create_collection_sparse(self, collection_name: str):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )
        else:
            print(f"Collection {collection_name} already exists.")
            
    def create_collection_hybrid(self, collection_name: str, embedding_dimensionality: int):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "jina-small": models.VectorParams(
                        size=embedding_dimensionality,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )
        else:
            print(f"Collection {collection_name} already exists.")