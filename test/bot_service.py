
import requests
import json
import time
from embeddings import find_most_relevant
from dotenv import load_dotenv
import os

# Ваш API ключ
load_dotenv()
API_KEY = os.getenv("API_KEY")
PROXY_URL = os.getenv("PROXY_URL")


def get_response_with_knowledge(user_input, knowledge_base):
    start_time = time.time()

    # Формируем промпт для модели с использованием базы знаний
    prompt = (
        "Ты — помощник, который отвечает на вопросы на основе предоставленной базы знаний. "
        "Вот контекст:\n\n"
        f"{knowledge_base}\n\n"
        "Вопрос:\n"
        f"{user_input}\n\n"
        "Ответь, используя только информацию из предоставленного контекста. "
        "Если в контексте нет ответа, скажи, что не знаешь."
    )

    # Данные для запроса
    data = {
        "model": "gpt-4-turbo",  # Указываем модель
        "messages": [
            {"role": "system", "content": prompt},
            #{"role": "user", "content": prompt},
        ],
        "max_tokens": 100,  # Максимальное количество токенов в ответе
        "temperature": 0.01,  # Параметр "творчества" модели
    }

    # Заголовки запроса
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Отправляем POST-запрос на прокси-сервер
    response = requests.post(PROXY_URL, headers=headers, json=data)

    # Проверяем, что запрос успешен
    if response.status_code == 200:
        result = response.json()
        end_time = time.time()
        print(f"Ответ с базой знаний занял: {end_time - start_time:.2f} секунд")
        return result["choices"][0]["message"]["content"].strip()
    else:
        end_time = time.time()
        print(f"Ответ с базой знаний занял: {end_time - start_time:.2f} секунд (ошибка)")
        return f"Ошибка: {response.status_code}, {response.text}"


def get_response_with_embedding(user_input, knowledge_base, knowledge_embedding):
    start_time = time.time()

    # Находим релевантные фрагменты
    relevant_knowledge = find_most_relevant(user_input, knowledge_base, knowledge_embedding)

    # Формируем промпт для модели с использованием релевантных фрагментов
    prompt = (
        "Ты — помощник, который отвечает на вопросы на основе предоставленной базы знаний. "
        "Вот контекст:\n\n"
        f"{relevant_knowledge if relevant_knowledge else knowledge_base}\n\n"
        "Вопрос:\n"
        f"{user_input}\n\n"
        "Ответь, используя только информацию из предоставленного контекста. "
        "Если в контексте нет ответа, скажи, что не знаешь."
    )

    # Данные для запроса
    data = {
        "model": "gpt-4-turbo",  # Указываем модель
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 100,  # Максимальное количество токенов в ответе
        "temperature": 0.7,  # Параметр "творчества" модели
    }

    # Заголовки запроса
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Отправляем POST-запрос на прокси-сервер
    response = requests.post(PROXY_URL, headers=headers, json=data)

    # Проверяем, что запрос успешен
    if response.status_code == 200:
        result = response.json()
        end_time = time.time()
        print(f"Ответ с использованием эмбеддинга занял: {end_time - start_time:.2f} секунд")
        return result["choices"][0]["message"]["content"].strip()
    else:
        end_time = time.time()
        print(f"Ответ с использованием эмбеддинга занял: {end_time - start_time:.2f} секунд (ошибка)")
        return f"Ошибка: {response.status_code}, {response.text}"


def get_response_without_knowledge(user_input):
    start_time = time.time()

    # Данные для запроса (без базы знаний)
    data = {
        "model": "gpt-4-turbo",  # Указываем модель
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ],
        "max_tokens": 100,  # Максимальное количество токенов в ответе
        "temperature": 0.5,  # Параметр "творчества" модели
    }

    # Заголовки запроса
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Отправляем POST-запрос на прокси-сервер
    response = requests.post(PROXY_URL, headers=headers, json=data)

    # Проверяем, что запрос успешен
    if response.status_code == 200:
        result = response.json()
        end_time = time.time()
        print(f"Ответ без базы знаний занял: {end_time - start_time:.2f} секунд")
        return result["choices"][0]["message"]["content"].strip()
    else:
        end_time = time.time()
        print(f"Ответ без базы знаний занял: {end_time - start_time:.2f} секунд (ошибка)")
        return f"Ошибка: {response.status_code}, {response.text}"