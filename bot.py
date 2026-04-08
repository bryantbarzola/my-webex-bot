"""
Webex AI Bot - Cisco Live Walk-In Lab
Build Your Own Personalized Webex AI Bot: Fast, Fun, and Surprisingly Powerful

This bot connects to Webex and responds to messages using an AI model.
Features: conversation memory, room restriction, custom personality.
"""

import os
import json
import requests
import boto3
from dotenv import load_dotenv
from webex_bot.webex_bot import WebexBot
from webex_bot.models.command import Command
from webex_bot.models.response import Response

load_dotenv()

# ---------------------------------------------------------
# CONFIG - Change these in your .env file
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
AI_PROVIDER = os.getenv("AI_PROVIDER")
AI_API_KEY = os.getenv("AI_API_KEY")

# Bedrock-specific config (only needed if AI_PROVIDER=bedrock)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ---------------------------------------------------------
# ROOM RESTRICTION
# ---------------------------------------------------------
# Set ALLOWED_ROOMS in your .env to lock the bot to specific
# Webex spaces. Leave it blank to allow all spaces.
#
# How to get a room ID:
#   1. Message your bot: "room info"
#   2. Copy the room ID from the response
#   3. Paste it into ALLOWED_ROOMS in your .env
#
# Multiple rooms: separate with commas (no spaces)
#   ALLOWED_ROOMS=roomId1,roomId2,roomId3
# ---------------------------------------------------------
_allowed_rooms_raw = os.getenv("ALLOWED_ROOMS", "")
ALLOWED_ROOMS = [r.strip() for r in _allowed_rooms_raw.split(",") if r.strip()]

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
# CONVERSATION MEMORY
# ---------------------------------------------------------
# The bot remembers what you said earlier in the conversation.
# Each user gets their own separate memory per room.
# We keep the last 20 messages (10 back-and-forth exchanges).
# ---------------------------------------------------------

MAX_MEMORY_MESSAGES = 20
conversations = {}


def get_memory_key(room_id: str, user_email: str) -> str:
    """Build a unique key per user per room."""
    return f"{room_id}_{user_email}"


def get_memory(key: str) -> list:
    """Return the conversation history for this key."""
    return conversations.get(key, [])


def add_to_memory(key: str, role: str, content: str):
    """Store a message and trim if over the limit."""
    if key not in conversations:
        conversations[key] = []
    conversations[key].append({"role": role, "content": content})
    if len(conversations[key]) > MAX_MEMORY_MESSAGES:
        conversations[key] = conversations[key][-MAX_MEMORY_MESSAGES:]


def clear_memory(key: str) -> bool:
    """Erase conversation history for this key."""
    if key in conversations:
        del conversations[key]
        return True
    return False


# ---------------------------------------------------------
# AI PROVIDER - This function handles all AI API calls
# ---------------------------------------------------------
def ask_ai(user_message: str, memory_key: str = None) -> str:
    """Send a message to the configured AI provider and return the response."""

    history = get_memory(memory_key) if memory_key else []

    if AI_PROVIDER == "openai":
        reply = _call_openai(user_message, history)
    elif AI_PROVIDER == "claude":
        reply = _call_claude(user_message, history)
    elif AI_PROVIDER == "bedrock":
        reply = _call_bedrock(user_message, history)
    else:
        return f"Unknown AI provider: {AI_PROVIDER}. Set AI_PROVIDER to 'openai', 'claude', or 'bedrock' in your .env file."

    if memory_key:
        add_to_memory(memory_key, "user", user_message)
        add_to_memory(memory_key, "assistant", reply)

    return reply


def _build_openai_messages(user_message: str, history: list) -> list:
    """Build the messages array for OpenAI-compatible APIs."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def _call_openai(user_message: str, history: list) -> str:
    """Call OpenAI API."""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {AI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "messages": _build_openai_messages(user_message, history),
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _call_claude(user_message: str, history: list) -> str:
    """Call Anthropic Claude API."""
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

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
            "messages": messages,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def _call_bedrock(user_message: str, history: list) -> str:
    """Call AWS Bedrock (Claude) API."""
    client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    })

    response = client.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


# ---------------------------------------------------------
# ROOM RESTRICTION HELPER
# ---------------------------------------------------------
def is_room_allowed(room_id: str) -> bool:
    """Check if the bot is allowed to respond in this room.
    If ALLOWED_ROOMS is empty, all rooms are allowed."""
    if not ALLOWED_ROOMS:
        return True
    return room_id in ALLOWED_ROOMS


# ---------------------------------------------------------
# BOT COMMANDS
# ---------------------------------------------------------
class RoomInfo(Command):
    def __init__(self):
        super().__init__(
            command_keyword="room info",
            help_message="Show the Webex room ID (use this to set up room restrictions)",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        room_id = activity["target"]["id"]

        response = Response()
        response.markdown = (
            f"**Room ID:** `{room_id}`\n\n"
            "To restrict the bot to this room, add this to your `.env` file:\n\n"
            f"```\nALLOWED_ROOMS={room_id}\n```\n\n"
            "Then restart the bot."
        )
        return response


class ClearMemory(Command):
    def __init__(self):
        super().__init__(
            command_keyword="clear memory",
            help_message="Clear your conversation history with TARS",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        sender = activity["actor"]["emailAddress"]
        room = activity["target"]["id"]
        key = get_memory_key(room, sender)

        response = Response()
        if clear_memory(key):
            response.markdown = "Memory wiped. Starting fresh, just like after a reboot."
        else:
            response.markdown = "Nothing to clear — your memory was already empty."
        return response


class AskTARS(Command):
    def __init__(self):
        super().__init__(
            command_keyword=" ",
            help_message="Talk to TARS - just type anything!",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        """Process any message sent to the bot."""
        room = activity["target"]["id"]

        if not is_room_allowed(room):
            return "Sorry, I'm not authorized to respond in this room."

        user_message = message.strip()
        if not user_message:
            return "I need something to work with, Cooper. Send me a message."

        sender = activity["actor"]["emailAddress"]
        memory_key = get_memory_key(room, sender)

        try:
            return ask_ai(user_message, memory_key)
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
        print("ERROR: AI_PROVIDER is missing. Set it to 'openai', 'claude', or 'bedrock' in your .env file.")
        exit(1)
    if AI_PROVIDER == "bedrock":
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            print("ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required for bedrock provider.")
            exit(1)
    elif not AI_API_KEY:
        print("ERROR: AI_API_KEY is missing. Add it to your .env file.")
        exit(1)

    print(f"Starting TARS bot with AI provider: {AI_PROVIDER}")
    print(f"Conversation memory: enabled (last {MAX_MEMORY_MESSAGES} messages per user)")
    if ALLOWED_ROOMS:
        print(f"Room restriction: enabled ({len(ALLOWED_ROOMS)} room(s) allowed)")
    else:
        print("Room restriction: disabled (responding in all rooms)")
    print("Press Ctrl+C to stop the bot.\n")

    bot = WebexBot(BOT_TOKEN, help_command=AskTARS())
    bot.add_command(RoomInfo())
    bot.add_command(ClearMemory())
    bot.run()
