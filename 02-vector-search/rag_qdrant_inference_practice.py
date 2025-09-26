#!/usr/bin/env python
# coding: utf-8

import os

from utility_functions import QdrantVectorStore
from qdrant_client import models
from openai import OpenAI
from config import QDRANT_HOST, COLLECTION_NAME, MODEL_HANDLE


def search(query, collection_name, model_handle, limit:int=1):
    results = client.query_points(collection_name=collection_name,
                                  query=models.Document(
                                      text=query,
                                      model=model_handle
                                  ),
                                  limit=limit,
                                  with_payload=True
                                 )
    return results.points



def search_by_filter(query, 
                     client,
                     filter_name: str='course',
                     filter_value: str='mlops-zoomcamp',
                     collection_name: str=COLLECTION_NAME,
                     model_handle: str=MODEL_HANDLE,
                     limit:int=1):
    results = client.query_points(collection_name=collection_name,
                                  query=models.Document(
                                      text=query,
                                      model=model_handle
                                  ),
                                  query_filter=models.Filter(
                                      must=[
                                          models.FieldCondition(
                                              key=filter_name,
                                              match=models.MatchValue(value=filter_value)
                                          )
                                      ]
                                  ),
                                  limit=limit,
                                  with_payload=True
                                 )
    return results.points


def search_sparse(query: str, client, collection_name: str, limit: int=1) -> list[models.ScoredPoint]:

    results = client.query_points(
        collection_name=collection_name,
        query=models.Document(
            text=query,
            model="Qdrant/bm25"
        ),
        limit=limit,
        with_payload=True,
        using="bm25"
    )
    return results

def search_multi_stage_sparse_and_dense(query: str, client, collection_name, limit: int) -> list[models.ScoredPoint]:
    
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="jinaai/jina-embeddings-v2-small-en"
                ),
                using='jina-small',
                limit=(limit*5)
            )
        ],
        query=models.Document(
            text=query,
            model="Qdrant/bm25"
        ),
        limit=limit,
        using="bm25",
        with_payload=True
    )
    return results.points

def search_hybrid(query: str, client, collection_name: str, limit: int=1) -> list[models.ScoredPoint]:
    
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="jinaai/jina-embeddings-v2-small-en"
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
    def __init__(self, api_key, model="gpt-4.1-nano"):
        """
        Initializes the OpenAI client with the provided API key.
        
        Args:
            api_key (str): OpenAI API key.
            model (str, optional): The model to use for chat completions. Defaults to "gpt-4.1-nano".
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, query):
        """
        Sends a chat request to the OpenAI API.
        
        Args:
            query (str): The query to send to the OpenAI API.

        Returns:
            Response from the OpenAI API.
        """
        response =  self.client.chat.completions.create(model=self.model,
                                                        messages=[{"role": "user", "content": query}]
                                                        )

        return response.choices[0].message.content

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
                context += f"Section: {doc.payload['section']}\nanswer: {doc.payload['text']}\n\n"
                course = doc.payload['course']
        else:
            doc = point.payload
            context += f"Section: {doc['section']}\nanswer: {doc['text']}\n\n"
            course = doc['course']
    context = f"Course: {course}" + "\n\n" + context
    return context


if __name__ == "__main__":
    
    # Initialize Qdrant client
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    qv = QdrantVectorStore(host=QDRANT_HOST)

    MODEL = "gpt-4.1-nano"
    prompt_template = """
    You are a course assistant, and your goal is to answer questions of students, where QUESTION is provided below and CONTEXT is provided most of the times. 
    Rules:
    * Answer the QUESTION based on the CONTEXT.
    * Use only the facts from the CONTEXT.
    * If CONTEXT is empty, please let the student know, the information about their query is not there, however you found the following information on the web, by searching the web

    QUESTION: {question}

    CONTEXT: {context}
    """.strip()


    query = "When can I start the course?"

    # Perform the search from Qdrant vector database
    results = search_by_filter(query=query, client=qv.client, limit=5)

    # Build the context from the search results
    context = build_context(results)

    promt = prompt_template.format(question=query, context=context)

    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=MODEL)

    # Send the prompt to the OpenAI API
    client_response = client.chat(query=promt)
    # Print the response from the OpenAI API
    print(query)
    print()
    print(client_response)

    query = "data engineer"
    results = search_sparse(query=query, collection_name="zoomcamo-rag-sparse", client=qv.client, limit=5)

    # Build the context from the search results
    context = build_context(results)

    promt = prompt_template.format(question=query, context=context)

    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=MODEL)

    # Send the prompt to the OpenAI API
    client_response = client.chat(query=promt)
    # Print the response from the OpenAI API
    print(query)
    print()
    print(client_response)

    query = "data engineer"
    results = search_multi_stage_sparse_and_dense(query=query, collection_name="zoomcamo-rag-sparse-dense", client=qv.client, limit=5)

    # Build the context from the search results
    context = build_context(results)

    promt = prompt_template.format(question=query, context=context)

    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=MODEL)

    # Send the prompt to the OpenAI API
    client_response = client.chat(query=promt)
    # Print the response from the OpenAI API
    print(query)
    print()
    print(client_response)

    query = "data engineer"
    results = search_hybrid(query=query, collection_name="zoomcamo-rag-sparse-dense", client=qv.client, limit=5)

    # Build the context from the search results
    context = build_context(results)

    promt = prompt_template.format(question=query, context=context)

    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=MODEL)

    # Send the prompt to the OpenAI API
    client_response = client.chat(query=promt)
    # Print the response from the OpenAI API
    print(query)
    print()
    print(client_response)
