#!/usr/bin/env python
# coding: utf-8

from utility_functions import QdrantVectorStore
from qdrant_client import models
from openai import OpenAI
from config import QDRANT_HOST, COLLECTION_NAME, MODEL_HANDLE, PROMPT_TEMPLATE, INSTRUCTIONS

def search_hybrid(query: str, client, collection_name: str, limit: int=1) -> list[models.ScoredPoint]:
    
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=MODEL_HANDLE
                ),
                using='jina-small',
                limit=(limit*5)
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25"
                ),
                using="bm25",
                limit=(limit*5)
            )
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        with_payload=True,
    )
    return results.points


class OpenAIClient:
    def __init__(self, model="gpt-5.4-mini"):
        """
        Initializes the OpenAI client with the provided API key.
        
        Args:
            api_key (str): OpenAI API key.
            model (str, optional): The model to use for chat completions. Defaults to "gpt-4.1-nano".
        """
        self.client = OpenAI()
        self.model = model

    def llm_call(self, prompt: str) -> str:
        """Make a call to the LLM and return the response text."""

        history = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.responses.create(
            model=self.model,
            input=history
        )
        return response.output_text

def build_context(result):
    """
    Builds a context string from the provided documents.
    
    Args:
        documents (list): List of documents to build the context from.
        
    Returns:
        str: A formatted string containing the context from the documents.
    """
    context = ""
    for point in result:
        if isinstance(point, tuple):
            for doc in point[1]:
                context += f"Section: {doc.payload['section']}\nquestion: {doc.payload['question']}\nanswer: {doc.payload['text']}\n\n"
                course = doc.payload['course']
        else:
            doc = point.payload
            context += f"Section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
            course = doc['course']
    context = f"Course: {course}" + "\n\n" + context
    return context


if __name__ == "__main__":

    query = "when can I start the course"
    qv = QdrantVectorStore(host=QDRANT_HOST)
    results = search_hybrid(query=query, collection_name=COLLECTION_NAME
                            ,client=qv.client, limit=5)
    
    for i, doc in enumerate(results):
        if isinstance(doc, object):
            print(f"ID={doc.id}, Score={doc.score}, Payload={doc.payload}")
            print()
    # Build the context from the search results
    context = build_context(results)

    prompt = PROMPT_TEMPLATE.format(question=query, context=context)

    client = OpenAIClient()

    # Send the prompt to the OpenAI API
    client_response = client.llm_call(prompt=prompt)
    # Print the response from the OpenAI API
    print(query)
    print()
    print(client_response)
