import os

COURSES_URL = "https://datatalks.club/faq/json/courses.json"
COURSES_FAQ_BASE_URL = "https://datatalks.club/faq"
INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''
PROMPT_TEMPLATE = """
<USER QUESTION>
{question}
</USER QUESTION>

<CONTEXT>
{context}
</CONTEXT>
"""

QDRANT_HOST = os.getenv("QDRANT_CLIENT", "http://localhost:6333")
EMBEDDING_DIMENSIONALITY = 512
MODEL_HANDLE = "jinaai/jina-embeddings-v2-small-en"
COLLECTION_NAME = "zoomcamo-rag-sparse-dense"
