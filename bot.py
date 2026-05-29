"""
Webex AI Bot - Cisco Live Walk-In Lab
Build Your Own Personalized Webex AI Bot: Fast, Fun, and Surprisingly Powerful

This bot connects to Webex and responds to messages using an AI model.
Features: conversation memory, room restriction, custom personality.
"""

import os
import sys
import json
import threading
import requests
from dotenv import load_dotenv
from webex_bot.webex_bot import WebexBot
from webex_bot.models.command import Command
from webex_bot.models.response import Response

try:
    import boto3
except ImportError:
    boto3 = None

load_dotenv()

# ---------------------------------------------------------
# CONFIG - Change these in your .env file
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
AI_PROVIDER = os.getenv("AI_PROVIDER")
AI_API_KEY = os.getenv("AI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL")

# Default models per provider (used when AI_MODEL is not set)
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "claude": "claude-sonnet-4-6-20250514",
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# Bedrock-specific config (only needed if AI_PROVIDER=bedrock)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize Bedrock client once (reused across all requests)
bedrock_client = None
if AI_PROVIDER == "bedrock" and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    if boto3 is None:
        print("ERROR: boto3 is required for bedrock provider. Install it: pip install boto3")
        sys.exit(1)
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

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
# KNOWLEDGE DIRECTORY
# ---------------------------------------------------------
# Drop .txt or .md files into the knowledge/ folder and the
# bot will use them as reference material when answering.
# ---------------------------------------------------------
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")


def load_knowledge() -> str:
    """Read all .txt and .md files from the knowledge/ directory."""
    if not os.path.isdir(KNOWLEDGE_DIR):
        return ""
    contents = []
    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(KNOWLEDGE_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                contents.append(f"### {filename}\n{f.read().strip()}")
    return "\n\n".join(contents)


KNOWLEDGE = load_knowledge()
if KNOWLEDGE:
    SYSTEM_PROMPT += f"\n\n## Reference Information\nUse the following knowledge to answer questions when relevant:\n\n{KNOWLEDGE}\n"

# ---------------------------------------------------------
# CONVERSATION MEMORY
# ---------------------------------------------------------
# The bot remembers what you said earlier in the conversation.
# Each user gets their own separate memory per room.
# We keep the last 20 messages (10 back-and-forth exchanges).
# ---------------------------------------------------------

MAX_MEMORY_MESSAGES = 20
conversations = {}
_memory_lock = threading.Lock()


def get_memory_key(room_id: str, user_email: str) -> str:
    """Build a unique key per user per room."""
    return f"{room_id}:{user_email}"


def get_memory(key: str) -> list:
    """Return a copy of the conversation history for this key."""
    with _memory_lock:
        return list(conversations.get(key, []))


def add_to_memory(key: str, role: str, content: str):
    """Store a message and trim if over the limit (keeping pairs intact)."""
    with _memory_lock:
        if key not in conversations:
            conversations[key] = []
        conversations[key].append({"role": role, "content": content})
        if len(conversations[key]) > MAX_MEMORY_MESSAGES:
            trimmed = conversations[key][-MAX_MEMORY_MESSAGES:]
            if trimmed and trimmed[0]["role"] == "assistant":
                trimmed = trimmed[1:]
            conversations[key] = trimmed


def clear_memory(key: str) -> bool:
    """Erase conversation history for this key."""
    with _memory_lock:
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


def _call_openai(user_message: str, history: list) -> str:
    """Call OpenAI API."""
    if not AI_API_KEY:
        raise ValueError("AI_API_KEY is not configured")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {AI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL or DEFAULT_MODELS["openai"],
            "messages": messages,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return "The AI returned an empty response. Try again."
    return choices[0]["message"]["content"]


def _call_claude(user_message: str, history: list) -> str:
    """Call Anthropic Claude API."""
    if not AI_API_KEY:
        raise ValueError("AI_API_KEY is not configured")
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
            "model": AI_MODEL or DEFAULT_MODELS["claude"],
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("content", [])
    if not content:
        return "The AI returned an empty response. Try again."
    return content[0]["text"]


def _call_bedrock(user_message: str, history: list) -> str:
    """Call AWS Bedrock (Claude) API."""
    if bedrock_client is None:
        raise ValueError("Bedrock client is not initialized. Check AWS credentials.")
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    })

    response = bedrock_client.invoke_model(
        modelId=AI_MODEL or DEFAULT_MODELS["bedrock"],
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    content = result.get("content", [])
    if not content:
        return "The AI returned an empty response. Try again."
    return content[0]["text"]


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
# WELCOME CARD - Adaptive Card shown when user types "help"
# ---------------------------------------------------------
WELCOME_CARD = {
    "type": "AdaptiveCard",
    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
    "version": "1.3",
    "body": [
        {
            "type": "TextBlock",
            "text": "TARS Bot",
            "weight": "Bolder",
            "size": "Large",
        },
        {
            "type": "TextBlock",
            "text": "AI assistant with 75% humor calibration. Type anything to chat with me, or use the quick actions below.",
            "wrap": True,
            "spacing": "Small",
        },
        {
            "type": "Input.Text",
            "id": "message",
            "placeholder": "Type a message for TARS...",
        },
    ],
    "actions": [
        {
            "type": "Action.Submit",
            "title": "Send to TARS",
            "data": {"callback_keyword": "help_card", "action": "chat"},
        },
        {
            "type": "Action.Submit",
            "title": "Clear Memory",
            "data": {"callback_keyword": "help_card", "action": "clear"},
        },
        {
            "type": "Action.Submit",
            "title": "Room Info",
            "data": {"callback_keyword": "help_card", "action": "room_info"},
        },
    ],
}


# ---------------------------------------------------------
# SHARED RESPONSES
# ---------------------------------------------------------
def _clear_memory_response(key: str) -> str:
    if clear_memory(key):
        return "Memory wiped. Starting fresh, just like after a reboot."
    return "Nothing to clear — your memory was already empty."


def _room_info_response(room_id: str):
    response = Response()
    response.markdown = (
        f"**Room ID:** `{room_id}`\n\n"
        "To restrict the bot to this room, add this to your `.env` file:\n\n"
        f"```\nALLOWED_ROOMS={room_id}\n```\n\n"
        "Then restart the bot."
    )
    return response


# ---------------------------------------------------------
# BOT COMMANDS
# ---------------------------------------------------------
class HelpCard(Command):
    def __init__(self):
        super().__init__(
            command_keyword="help_card",
            help_message="Handle welcome card button clicks",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        """Handle button clicks from the welcome card."""
        action = attachment_actions.inputs.get("action", "")
        sender = activity["actor"]["emailAddress"]
        room = activity["target"]["id"]

        if action == "chat":
            user_message = attachment_actions.inputs.get("message", "").strip()
            if not user_message:
                return "Type a message in the text box first, Cooper."
            if not is_room_allowed(room):
                return "Sorry, I'm not authorized to respond in this room."
            memory_key = get_memory_key(room, sender)
            try:
                return ask_ai(user_message, memory_key)
            except Exception:
                return "Something went wrong talking to the AI. Try again in a moment."

        elif action == "clear":
            key = get_memory_key(room, sender)
            return _clear_memory_response(key)

        elif action == "room_info":
            return _room_info_response(room)

        return "Unknown action. Try again!"


class Help(Command):
    def __init__(self):
        super().__init__(
            command_keyword="help",
            help_message="Show the TARS welcome card with quick actions",
            card=WELCOME_CARD,
        )
        self.card_callback_keyword = None

    def execute(self, message, attachment_actions, activity):
        return ""


class RoomInfo(Command):
    def __init__(self):
        super().__init__(
            command_keyword="room info",
            help_message="Show the Webex room ID (use this to set up room restrictions)",
            card=None,
        )

    def execute(self, message, attachment_actions, activity):
        return _room_info_response(activity["target"]["id"])


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
        return _clear_memory_response(key)


class AskTARS(Command):
    def __init__(self):
        super().__init__(
            command_keyword="",
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
        except Exception:
            return "Something went wrong talking to the AI. Try again in a moment."


# ---------------------------------------------------------
# START THE BOT
# ---------------------------------------------------------
if __name__ == "__main__":
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN is missing. Add it to your .env file.")
        sys.exit(1)
    if not AI_PROVIDER:
        print("ERROR: AI_PROVIDER is missing. Set it to 'openai', 'claude', or 'bedrock' in your .env file.")
        sys.exit(1)
    if AI_PROVIDER == "bedrock":
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            print("ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required for bedrock provider.")
            sys.exit(1)
    elif not AI_API_KEY:
        print("ERROR: AI_API_KEY is missing. Add it to your .env file.")
        sys.exit(1)

    model = AI_MODEL or DEFAULT_MODELS.get(AI_PROVIDER, "unknown")
    print(f"Starting TARS bot with AI provider: {AI_PROVIDER} (model: {model})")
    print(f"Conversation memory: enabled (last {MAX_MEMORY_MESSAGES} messages per user)")
    if ALLOWED_ROOMS:
        print(f"Room restriction: enabled ({len(ALLOWED_ROOMS)} room(s) allowed)")
    else:
        print("Room restriction: disabled (responding in all rooms)")
    if KNOWLEDGE:
        file_count = KNOWLEDGE.count("### ")
        print(f"Knowledge files: {file_count} loaded from knowledge/")
    else:
        print("Knowledge files: none (add .txt or .md files to knowledge/ to give the bot reference info)")
    print("Press Ctrl+C to stop the bot.\n")

    bot = WebexBot(BOT_TOKEN, help_command=AskTARS())
    bot.add_command(Help())
    bot.add_command(HelpCard())
    bot.add_command(RoomInfo())
    bot.add_command(ClearMemory())
    bot.run()
