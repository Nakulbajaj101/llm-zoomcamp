#!/usr/bin/env python
# coding: utf-8
import json
import os

import minsearch
from openai import OpenAI


def extract_documents(files: list[str]) -> list:
    """
    Extracts documents from a list of JSON files.
    
    Args:
        files (list[str]): List of file paths to JSON files containing documents.
        
    Returns:
        list: A list of documents extracted from the provided files.
    """
    documents = []
    for file in files:
        with open(f"../01-intro/{file}", "rt") as f_in:
            docs_raw = json.load(f_in)
        for course_dict in docs_raw:
            for doc in course_dict['documents']:
                doc['course'] = course_dict['course']
                documents.append(doc)
    return documents


class Minsearch:
    def __init__(self, text_fields, keyword_fields):
        """
        Initializes the Minsearch index with specified text and keyword fields.
        
        Args:
            text_fields (list): List of text fields to be indexed.
            keyword_fields (list): List of keyword fields to be indexed.
        """
        self.index = minsearch.Index(text_fields=text_fields, keyword_fields=keyword_fields)

    def fit(self, documents):
        """
        Fits the index with the provided documents.
        
        Args:
            documents (list): List of documents to be indexed.
        """
        self.index.fit(documents)

    def search(self, query, boost_dict=None, filter_dict=None, num_results=10):
        """
        Searches the index for the given query with optional boosting and filtering.
        
        Args:
            query (str): The search query.
            boost_dict (dict, optional): Dictionary of fields to boost in the search.
            filter_dict (dict, optional): Dictionary of fields to filter results.
            num_results (int, optional): Number of results to return. Defaults to 10.
        
        Returns:
            list: List of search results.
        """
        return self.index.search(query=query, boost_dict=boost_dict, filter_dict=filter_dict, num_results=num_results)



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

def build_context(results):
    """
    Builds a context string from the provided documents.
    
    Args:
        documents (list): List of documents to build the context from.
        
    Returns:
        str: A formatted string containing the context from the documents.
    """
    context = ""
    for doc in results:
        context += f"Section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
        course = doc['course']
    context = f"Course: {course}" + "\n\n" + context
    return context


if __name__ == "__main__":
    
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


    files = ["documents.json", "documents-llm.json"]
    documents = extract_documents(files)

    # Create the index
    index = Minsearch(text_fields=["question", "text", "section"], keyword_fields=["course"])
    # Fit the index with the documents
    index.fit(documents)

    boost_dict = {"question": 3.0, "section": 0.3}
    filter_dict = {"course": "data-engineering-zoomcamp"}

    query = "When can I start the course?"
    # Search the index
    results = index.search(query=query,
                           boost_dict=boost_dict,
                           filter_dict=filter_dict, num_results=5)

    # Build the context from the search results
    context = build_context(results)

    promt = prompt_template.format(question=query, context=context)

    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=MODEL)

    # Send the prompt to the OpenAI API
    client_response = client.chat(query=promt)

    print(client_response)
