from openai import OpenAI

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