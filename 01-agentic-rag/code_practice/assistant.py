from openai import OpenAI
from minsearch import Index
from documents_indexing import create_index
from data_ingestion import fetch_data, fetch_course_faqs
from config import COURSES_URL, INSTRUCTIONS, PROMPT_TEMPLATE


class FaqRag:
    def __init__(self, index: Index, prompt_template: str, system_instruction: str = "", model_name: str = "gpt-5.4-mini"):
        self.index = index
        self.prompt_template = prompt_template
        self.system_instruction = system_instruction
        self.model_name = "gpt-5.4-mini" if not model_name else model_name
        self._client = OpenAI()

    @staticmethod
    def build_context(results: list) -> str:
        docs = []
        for result in results:
            docs.append(f"Section: {result['section']}")
            docs.append(f"Question: {result['question']}")
            docs.append(f"Answer: {result['answer']}")
            docs.append(" ")
        return "\n".join(docs).strip()

    def search_results(self, question: str, filters: dict, num_results: int=5) -> list:
        results = self.index.search(query=question,
                boost_dict={"question": 2.0, "section" : 0.5},
                filter_dict=filters,
                num_results=num_results)

        return results
    
    def llm_call(self, prompt: str) -> str:
        """Make a call to the LLM and return the response text."""

        history = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        response = self._client.responses.create(
            model=self.model_name,
            input=history
        )
        return response.output_text
    
    def rag(self, question: str, filters: dict, num_results: int=5) -> str:
        results = self.search_results(question, filters, num_results)
        context = self.build_context(results)
        prompt = self.prompt_template.format(context=context, question=question)
        response = self.llm_call(prompt)
        return response
    
def create_faq_rag(text_fields: list, keyword_fields: str) -> FaqRag:
    courses = fetch_data(COURSES_URL)
    documents = fetch_course_faqs(courses)
    print(f"Fetched {len(documents)} FAQ entries.")
    
    index = create_index(documents, text_fields, keyword_fields)
    rag = FaqRag(
        index=index,
        prompt_template=PROMPT_TEMPLATE,
        system_instruction=INSTRUCTIONS
    )
    return rag

if __name__ == "__main__":
    text_fields = ["question", "answer", "section"]
    keyword_fields = ["course"]
    rag = create_faq_rag(text_fields, keyword_fields)

    user_query_response = rag.rag(
        question="I just discovered the course. Can I join?",
        filters={"course": "data-engineering-zoomcamp"},
        num_results=5)
    
    print(user_query_response)
