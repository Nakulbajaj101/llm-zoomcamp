import requests
from tqdm.auto import tqdm
from config import COURSES_URL, COURSES_FAQ_BASE_URL


def fetch_data(url: str) -> list[dict]:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_course_faqs(courses: dict) -> list[dict]:
    documents = []
    for course in tqdm(courses, desc="Fetching FAQs for courses"):
        url = f"{COURSES_FAQ_BASE_URL}{course['path']}"
        course_data = fetch_data(url)
        documents.extend(course_data)
    return documents

if __name__ == "__main__":
    courses = fetch_data(COURSES_URL)
    documents = fetch_course_faqs(courses)
    print(f"Fetched {len(documents)} FAQ entries.")
