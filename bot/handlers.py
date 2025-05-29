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
from app.quiz import quiz_pipeline, rate_answer_pipeline
from app.rag import rag_pipeline
from configs import load_config

filterwarnings(
    action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning
)
logger = setup_logger(__name__)

QUIT_QUIZ_MODE_MESSAGE = "Quit quiz mode"


class State(Enum):
    READY = 1
    QUIZ_SECTION_SELECTION = 2
    QUIZ_WAITING_ANSWER = 3


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
    context.user_data["quiz_question"] = None
    context.user_data["current_context"] = None

    sections = get_all_sections()
    sections.append(QUIT_QUIZ_MODE_MESSAGE)
    keyboard = [
        [InlineKeyboardButton(section, callback_data=section)] for section in sections
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_reply_text = "🌈 Choose the section"
    await update.message.reply_text(message_reply_text, reply_markup=reply_markup)
    return State.QUIZ_SECTION_SELECTION


async def section_choice_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == QUIT_QUIZ_MODE_MESSAGE:  # quit quiz mode option
        log_tg(update, "QUITTING QUIZ MODE")
        await query.edit_message_text(
            "Enter your question or write /quiz for a training"
        )
        return State.READY

    await query.edit_message_text("⌛")
    log_tg(update, f"SECTION {query.data}")

    response = quiz_pipeline(query.data)
    if not response["success"]:
        await query.edit_message_text(f"💥 {response['error']}")
        return State.READY

    quiz_question, current_context = response["question"], response["context"]
    await query.edit_message_text(
        f"❓ Question:\n{quiz_question}\n\nAnswer the question for me to rate it or write /answer if you want to learn it"
    )
    log_tg(update, f"MODEL QUESTION: {quiz_question}")
    context.user_data["quiz_question"] = quiz_question
    context.user_data["current_context"] = current_context
    return State.QUIZ_WAITING_ANSWER


async def handle_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    log_tg(update, "USER ANSWER")
    text = update.message.text.strip()
    current_context = context.user_data.get("current_context", None)
    question = context.user_data["quiz_question"]

    if text == "/answer":  # model answers the question
        sent_message = await update.message.reply_text("⌛")
        response = rag_pipeline(question, current_context)

        if not response["success"]:
            await sent_message.edit_text(f"💥 {response['error']}")
            return State.READY

        answer = response["answer"]
        await sent_message.edit_text(f"💡 Answer:\n{answer}")
        log_tg(update, f"MODEL ANSWER: {answer}")
        return await handle_quiz(update, context)

    sent_message = await update.message.reply_text("⌛")
    response = rate_answer_pipeline(  # user answered the question
        context=current_context, question=question, answer=text
    )

    if not response["success"]:
        await sent_message.edit_text(f"💥 {response['error']}")
        return State.READY

    evaluation = response["evaluation"]
    log_tg(update, f"USER ANSWER: {text}")
    log_tg(update, f"ANSWER EVALUATION: {evaluation}")
    await sent_message.edit_text(f"👍 {evaluation}")
    return await handle_quiz(update, context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    log_tg(update, f"USER'S QUERY: {query}")

    sent_message = await update.message.reply_text("⌛")
    response = rag_pipeline(query, None)

    if not response["success"]:
        await sent_message.edit_text(f"💥 {response['error']}")
        context.user_data["quiz_question"] = None
        context.user_data["current_context"] = None
        return State.READY

    answer = response["answer"]
    await sent_message.edit_text(f"💡 Answer:\n{answer}")
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
            State.QUIZ_SECTION_SELECTION: [CallbackQueryHandler(section_choice_button)],
            State.QUIZ_WAITING_ANSWER: [
                MessageHandler(filters.TEXT, handle_quiz_answer)
            ],
            State.READY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
            ],
        },
        fallbacks=[
            CommandHandler("start", start),
            CommandHandler("quiz", handle_quiz),
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
        ],
        per_message=False,
    )

    app.add_handler(conv_handler)
    logger.info("BOT STARTED")
    app.run_polling()
