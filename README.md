# Webex AI Bot

A simple, single-file Webex bot powered by AI. Built for the Cisco Live walk-in lab: *Build Your Own Personalized Webex AI Bot: Fast, Fun, and Surprisingly Powerful*.

## Features

- **Multi-provider AI** — supports OpenAI, Anthropic Claude, and AWS Bedrock
- **Conversation memory** — remembers the last 20 messages per user per room
- **Custom personality** — change the system prompt to make the bot act however you want
- **Room restriction** — lock the bot to specific Webex spaces
- **Adaptive Card** — type "help" for a welcome card with quick actions

## Quick Start

### 1. Create a Webex Bot

1. Go to [developer.webex.com](https://developer.webex.com/) and sign in
2. Profile icon → **My Webex Apps** → **Create a New App** → **Create a Bot**
3. Fill in name/username, click **Add Bot**
4. Copy the **Bot Access Token**

### 2. Configure

```bash
cp .env.sample .env
```

Edit `.env` and paste your bot token. Choose an AI provider and add its credentials:

| Provider | `AI_PROVIDER` | Auth needed |
|----------|---------------|-------------|
| AWS Bedrock | `bedrock` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| OpenAI | `openai` | `AI_API_KEY` |
| Anthropic | `claude` | `AI_API_KEY` |

### 3. Install & Run

```bash
pip install -r requirements.txt
python bot.py
```

### 4. Talk to it

Open [web.webex.com](https://web.webex.com/), search for your bot's username, and send it a message.

## Bot Commands

| Command | What it does |
|---------|-------------|
| `help` | Show welcome card with quick actions |
| `clear memory` | Wipe your conversation history |
| `room info` | Show the current room ID (for room restriction) |

## Customize the Personality

Find `SYSTEM_PROMPT` near the top of `bot.py` and change it to whatever you want. Restart the bot to apply.

## File Structure

```
bot.py            # The entire bot (single file, ~390 lines)
.env.sample       # Configuration template
requirements.txt  # Python dependencies
```

## Requirements

- Python 3.9+
- A Webex bot token
- An AI provider API key (or AWS credentials for Bedrock)

## License

MIT
