import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import logging
from src.graph_model import graph, retrieve
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

async def start(update: Update, context):
    await update.message.reply_text(
        "Салам, я бот, созданный для тестирования проекта. Если тоже хочешь тестировать - дерзай!"
    )


async def handle_message(update: Update, context):
    user_input = update.message.text

    response = None
    for step in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
        config={"configurable": {"thread_id": "439164"}},
    ):
        response = step["messages"][-1].content

    # Отправляем ответ пользователю
    await update.message.reply_text(response)


if __name__ == "__main__":

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()