import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Your Telegram Bot token
TELEGRAM_TOKEN = "YOU CAN GET THIS TOKEN FROM BotFather BOT"

# Your Flask server (make sure it's running!)
API_URL = "http://127.0.0.1:5000/ask"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("I’m your AI bot, ask me anything!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_q = update.message.text

    # Check for shutdown command
    if user_q.strip() == "SHUTERROR:44":
        try:
            res = requests.post("http://127.0.0.1:5000/shutdown")
            if res.status_code == 200:
                await update.message.reply_text("Server is shutting down. Bot will stop now.")
                # Stop the bot gracefully
                context.application.stop()
                return
            else:
                await update.message.reply_text("Failed to shut down the server.")
        except Exception as e:
            await update.message.reply_text(f"Error during shutdown: {e}")
        return

    try:
        res = requests.post(API_URL, json={"query": user_q})
        data = res.json()
        answer = data.get("answer", "Bro, I blanked out 😭")
    except Exception as e:
        answer = f"Oops, server trippin’: {e}"

    await update.message.reply_text(answer)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 Telegram bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
