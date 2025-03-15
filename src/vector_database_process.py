import os
import json
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("PROXY_URL"),
)

VECTOR_STORE_PATH = "src/vector_store.faiss"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

def parse_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    page_content = soup.get_text(separator=" ", strip=True)
    return page_content

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECTIONS_PATH = os.path.join(BASE_DIR, "sections.json")

def load_sections(file_path: str) -> list[dict]:
    """Загружает данные о разделах из JSON-файла."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден!")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def create_documents(sections: list[dict]) -> list[Document]:
    """Парсит страницы и создаёт документы."""
    documents = []
    for section in sections:
        url = section["url"]
        title = section["title"]
        page_content = parse_page(url)
        chunks = text_splitter.split_text(page_content)
        for i, chunk in enumerate(chunks):
            document = Document(
                page_content=chunk,
                metadata={
                    "source": "website",
                    "title": title,
                    "url": url,
                    "chunk_id": i + 1,
                }
            )
            documents.append(document)
    return documents

def load_or_create_vector_store(docs: list[Document], embeddings, path: str):
    """Загружает векторную базу данных из файла или создаёт её, если файла нет."""
    if os.path.exists(path):
        print("Загрузка векторной базы данных из файла...")
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Создание новой векторной базы данных...")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(path)
        return vector_store


sections = load_sections(SECTIONS_PATH)

documents = create_documents(sections)

vector_store = load_or_create_vector_store(documents, embeddings, VECTOR_STORE_PATH)

__all__ = ["vector_store", "embeddings"]