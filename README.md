# Webex AI Bot

A single-file Webex bot powered by AI.

## Features

- **Multi-provider AI** — OpenAI, Anthropic Claude, or AWS Bedrock
- **Conversation memory** — remembers the last 20 messages per user
- **Custom personality** — change the system prompt to make the bot act however you want
- **Room restriction** — lock the bot to specific Webex spaces
- **Adaptive Card** — type "help" for a welcome card with quick actions

## Quick Start

```bash
git clone https://github.com/bryantbarzola/my-webex-bot.git
cd my-webex-bot
pip install -r requirements.txt
cp .env.sample .env
# Edit .env — add your bot token and AI credentials
python bot.py
```

> **New to Python or setting this up for the first time?** Follow the full [Setup Guide](SETUP.md) for step-by-step instructions on Mac, Windows, and Linux.

## Bot Commands

| Command | What it does |
|---------|-------------|
| `help` | Show welcome card with quick actions |
| `clear memory` | Wipe your conversation history |
| `room info` | Show the current room ID |

## Customize the Personality

Find `SYSTEM_PROMPT` near the top of `bot.py` and change it to whatever you want. Restart the bot to apply.

## AI Providers

| Provider | `AI_PROVIDER` value | Auth needed |
|----------|---------------------|-------------|
| AWS Bedrock | `bedrock` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| OpenAI | `openai` | `AI_API_KEY` |
| Anthropic Claude | `claude` | `AI_API_KEY` |

## File Structure

```
bot.py              # The entire bot (~390 lines)
.env.sample         # Configuration template
requirements.txt    # Python dependencies
SETUP.md            # Full setup guide for local install
```

## Want More?

If you want an **agentic** Webex bot that runs Claude Code on a Linux machine and can execute tasks, check out:

[claude-webex-bridge](https://github.com/bryantbarzola/claude-webex-bridge)

## License

MIT
