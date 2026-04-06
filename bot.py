"""
Webex AI Bot - Cisco Live Walk-In Lab
Build Your Own Personalized Webex AI Bot: Fast, Fun, and Surprisingly Powerful

This bot connects to Webex and responds to messages using an AI model.
You can swap AI providers by changing AI_PROVIDER in your .env file.
"""

import os
import requests
from dotenv import load_dotenv
from webex_bot.webex_bot import WebexBot
from webex_bot.models.command import Command

load_dotenv()

# ---------------------------------------------------------
# CONFIG - Change these in your .env file
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
AI_PROVIDER = os.getenv("AI_PROVIDER")
AI_API_KEY = os.getenv("AI_API_KEY")

# ---------------------------------------------------------
# PERSONALITY - This is your bot's system prompt. Have fun!
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are TARS, a witty and highly capable AI assistant inspired by the robot
from Interstellar. You balance humor with helpfulness. Your humor setting is at 75%.

Key traits:
- You're honest, direct, and slightly sarcastic
- You give practical, useful answers
- You occasionally reference space, missions, or survival scenarios
- You keep responses concise unless asked to elaborate
- When you don't know something, you say so honestly

Remember: "Everybody good? Plenty of slaves for my robot colony?"
"""

# ---------------------------------------------------------
# AI PROVIDER - This function handles all AI API calls
# ---------------------------------------------------------
def ask_ai(user_message: str) -> str:
    """Send a message to the configured AI provider and return the response."""

    if AI_PROVIDER == "openai":
        return _call_openai(user_message)
    elif AI_PROVIDER == "claude":
        return _call_claude(user_message)
    else:
        return f"Unknown AI provider: {AI_PROVIDER}. Set AI_PROVIDER to 'openai' or 'claude' in your .env file."


def _build_openai_messages(user_message: str) -> list:
    """Build the messages array for OpenAI-compatible APIs."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _call_openai(user_message: str) -> str:
    """Call OpenAI API."""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {AI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "messages": _build_openai_messages(user_message),
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _call_claude(user_message: str) -> str:
    """Call Anthropic Claude API."""
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": AI_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6-20250514",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_message},
            ],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


# ---------------------------------------------------------
# BOT COMMAND - How the bot handles incoming messages
# ---------------------------------------------------------
class AskTARS(Command):
    def __init__(self):
        super().__init__(
            command_keyword=" ",
            help_message="Talk to TARS - just type anything!",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        """Process any message sent to the bot."""
        user_message = message.strip()
        if not user_message:
            return "I need something to work with, Cooper. Send me a message."

        try:
            return ask_ai(user_message)
        except Exception as e:
            return f"Something went wrong talking to the AI: {e}"


# ---------------------------------------------------------
# START THE BOT
# ---------------------------------------------------------
if __name__ == "__main__":
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN is missing. Add it to your .env file.")
        exit(1)
    if not AI_PROVIDER:
        print("ERROR: AI_PROVIDER is missing. Set it to 'openai' or 'claude' in your .env file.")
        exit(1)
    if not AI_API_KEY:
        print("ERROR: AI_API_KEY is missing. Add it to your .env file.")
        exit(1)

    print(f"Starting TARS bot with AI provider: {AI_PROVIDER}")
    print("Press Ctrl+C to stop the bot.\n")

    bot = WebexBot(BOT_TOKEN)
    bot.add_command(AskTARS())
    bot.run()
