import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Используем FAISS для сохранения на диск
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from typing import List, TypedDict
from langgraph.graph import END
import requests
from bs4 import BeautifulSoup
import json

# Загрузка переменных окружения
load_dotenv()
API_KEY = os.getenv("API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Настройка окружения для LangSmith и OpenAI
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Инициализация эмбеддингов и языковой модели
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://api.proxyapi.ru/openai/v1",  # Указываем прокси
)

llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://api.proxyapi.ru/openai/v1",  # Указываем прокси
    temperature=0.01
)

# Путь для сохранения векторной базы данных
VECTOR_STORE_PATH = "src/vector_store.faiss"

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


# Загрузка данных из sections.json
def load_sections(file_path: str) -> List[dict]:
    """Загружает данные о разделах из JSON-файла."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# Создание документов и разбиение на chunks
def create_documents(sections: List[dict]) -> List[Document]:
    """Парсит страницы и создаёт документы."""
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
    return documents


# Загрузка или создание векторной базы данных
def load_or_create_vector_store(docs: List[Document], embeddings, path: str):
    """Загружает векторную базу данных из файла или создаёт её, если файла нет."""
    if os.path.exists(path):
        print("Загрузка векторной базы данных из файла...")
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Создание новой векторной базы данных...")
        # Парсим страницы и создаём документы только если база данных отсутствует
        sections = load_sections("src/sections.json")
        documents = create_documents(sections)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(path)
        return vector_store


# # Загрузка разделов
# sections = load_sections("src/sections.json")
#
# # Создание документов
# documents = create_documents(sections)

# Загрузка или создание векторной базы данных
vector_store = load_or_create_vector_store([], embeddings, VECTOR_STORE_PATH)


# Определение инструмента для поиска информации
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Ищет информацию, связанную с запросом, в векторном хранилище."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Определение узлов графа
graph_builder = StateGraph(MessagesState)


def query_or_respond(state: MessagesState):
    """Генерирует запрос к языковой модели или ответ."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """Генерирует ответ на основе найденной информации."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "Provide a detailed and comprehensive response, explaining each step clearly. "
        "If you don't know the answer, say that you don't know. "
        "Your response should be at least 5-7 sentences long."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
           or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


# Сборка графа
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# Запуск графа
# input_message = "What is Tree of Thoughts?"
#
# for step in graph.stream(
#         {"messages": [{"role": "user", "content": input_message}]},
#         stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
__all__ = ["graph", "vector_store", "retrieve"]
