import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import logging
from src.vector_database_process import graph  # Импортируем граф из vector_database_process.py

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Обработчик команды /start
async def start(update: Update, context):
    await update.message.reply_text(
        "Привет! Я бот, который может отвечать на вопросы. Просто напишите мне ваш вопрос, и я постараюсь помочь."
    )

# Обработчик текстовых сообщений
async def handle_message(update: Update, context):
    user_input = update.message.text  # Получаем текст сообщения от пользователя

    try:
        # Запускаем граф для обработки запроса
        response = None
        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
        ):
            response = step["messages"][-1].content  # Получаем последний ответ

        # Отправляем ответ пользователю
        await update.message.reply_text(response)

    except Exception as e:
        # Обработка ошибок
        logging.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text("Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.")

# Запуск бота
if __name__ == "__main__":
    # Создаем приложение для бота
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Регистрируем обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота
    application.run_polling()