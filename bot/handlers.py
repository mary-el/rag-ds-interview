from enum import Enum
from warnings import filterwarnings

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from telegram.warnings import PTBUserWarning

from app.database import get_all_sections
from app.logger import setup_logger
from app.quiz import quiz_pipeline
from app.rag import rag_pipeline
from configs import load_config

filterwarnings(
    action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning
)
logger = setup_logger(__name__)


class State(Enum):
    READY = 1
    QUIZ = 2


def log_tg(update: Update, message: str):
    user = update.effective_user
    user_id = user.id
    username = user.username or f"{user.first_name} {user.last_name}"
    logger.info(f"USER {username} ({user_id}): {message}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I will help you to preprare for the Data Scientist interview. "
        "Enter your question or write /quiz for a training"
    )

    context.user_data["quiz_question"] = None
    context.user_data["current_context"] = None

    log_tg(update, "START")
    return State.READY


async def handle_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log_tg(update, "QUIZ MODE")

    sections = get_all_sections()
    keyboard = [
        [InlineKeyboardButton(section, callback_data=section)] for section in sections
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_reply_text = "Choose the section"
    await update.message.reply_text(message_reply_text, reply_markup=reply_markup)
    return State.QUIZ


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    log_tg(update, f"SECTION {query.data}")

    quiz_question, current_context = quiz_pipeline(query.data)
    await query.edit_message_text(
        f"❓ Question:\n{quiz_question}\n\nWrite /answer or ask a new question"
    )
    log_tg(update, f"MODEL QUESTION: {quiz_question}")
    context.user_data["quiz_question"] = quiz_question
    context.user_data["current_context"] = current_context
    return State.READY


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    log_tg(update, f"USER'S QUERY: {text}")

    if text == "/answer":
        if context.user_data["quiz_question"] is not None:
            query = context.user_data["quiz_question"]
        else:
            await update.message.reply_text("You are not in /quiz mode")
            return State.READY
    else:
        query = text

    answer = rag_pipeline(query, context.user_data.get("current_context", None))
    await update.message.reply_text(f"💡 Answer:\n{answer}")
    log_tg(update, f"MODEL ANSWER: {answer}")

    context.user_data["quiz_question"] = None
    context.user_data["current_context"] = None
    return State.READY


def run_bot():
    config = load_config()["interface"]
    app = ApplicationBuilder().token(config["telegram_token"]).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            State.QUIZ: [CallbackQueryHandler(button)],
            State.READY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
            ],
        },
        fallbacks=[
            CommandHandler("start", start),
            CommandHandler("answer", handle_message),
            CommandHandler("quiz", handle_quiz),
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
        ],
        per_message=False,
    )

    app.add_handler(conv_handler)
    logger.info("BOT STARTED")
    app.run_polling()
