import json
import requests
from bs4 import BeautifulSoup
from uuid import uuid4
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document

# Загрузка данных из sections.json
with open("../src/sections.json", "r", encoding="utf-8") as file:
    sections = json.load(file)

# Инициализация RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Размер каждого chunk (в символах)
    chunk_overlap=50,  # Перекрытие между chunks
    length_function=len,  # Функция для вычисления длины текста
    is_separator_regex=False,  # Не использовать регулярные выражения для разделителей
)


# Функция для парсинга страницы
def parse_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Извлечение текста страницы
    page_content = soup.get_text(separator=" ", strip=True)
    return page_content


# Создание документов и разбиение на chunks
documents = []
for section in sections:
    url = section["url"]
    title = section["title"]

    # Парсим страницу
    page_content = parse_page(url)

    # Разбиваем текст на chunks
    chunks = text_splitter.split_text(page_content)

    # Создаем объекты Document для каждого chunk
    for i, chunk in enumerate(chunks):
        document = Document(
            page_content=chunk,
            metadata={
                "source": "website",
                "title": title,
                "url": url,
                "chunk_id": i + 1,  # Номер chunk
            }
        )
        documents.append(document)

# Генерация уникальных идентификаторов
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)


print(f"Добавлено {len(documents)} chunks в векторное хранилище.")
