from minsearch import Index
from data_ingestion import fetch_data, fetch_course_faqs
from config import COURSES_URL

def create_index(documents: list[dict],
                 text_fields: list,
                 keyword_fields: list) -> Index:

    index = Index(
        text_fields = text_fields,
        keyword_fields = keyword_fields
        )
    index.fit(documents)
    return index

if __name__ == "__main__":
    courses = fetch_data(COURSES_URL)
    documents = fetch_course_faqs(courses)
    print(f"Fetched {len(documents)} FAQ entries.")
    
    text_fields = ["question", "answer", "section"]
    keyword_fields = ["course"]
    index = create_index(documents, text_fields, keyword_fields)
