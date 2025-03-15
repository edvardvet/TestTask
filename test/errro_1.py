import getpass
import os

from click import prompt
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain import hub
from typing_extensions import Annotated, List, TypedDict
from typing import Literal
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()
API_KEY = os.getenv("API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://api.proxyapi.ru/openai/v1",  # Указываем прокси
)

# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://api.proxyapi.ru/openai/v1",  # Указываем прокси
)


vector_store = InMemoryVectorStore(embeddings)


# Загрузка документов из файла all_documents.txt
def load_documents_from_file(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("Document"):
                # Пропускаем строку с номером документа
                continue
            if line.startswith("  Content:"):
                content = line.replace("  Content: ", "").strip()
            elif line.startswith("  Metadata:"):
                metadata = line.replace("  Metadata: ", "").strip()
                metadata = eval(metadata)  # Преобразуем строку в словарь
                # Создаем объект Document
                document = Document(page_content=content, metadata=metadata)
                documents.append(document)
    return documents

# Загрузка документов
docs = load_documents_from_file("all_documents.txt")


# Разбиение на чанки (если нужно)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"


# Index chunks
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)


# Define schema for search
class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

for step in graph.stream(
    {"question": "What is Tree of Thoughts?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")





