# Setup Guide

Full step-by-step instructions for running the Webex AI Bot on your own computer.

## Prerequisites

- A [Webex](https://web.webex.com/) account (free)
- An API key from one of: OpenAI, Anthropic, or AWS (Bedrock)

## Step 1: Install Python

You need Python 3.9 or newer.

### Mac

Python 3 comes pre-installed on modern macOS. Check by opening **Terminal** and running:

```bash
python3 --version
```

If you don't have it, install via [Homebrew](https://brew.sh/):

```bash
brew install python
```

### Windows

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the latest Python 3 installer
3. Run it — **check "Add python.exe to PATH"** during install
4. Open **Command Prompt** or **PowerShell** and verify:

```bash
python --version
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv
```

---

## Step 2: Clone the Repo

```bash
git clone https://github.com/bryantbarzola/my-webex-bot.git
cd my-webex-bot
```

If you don't have git installed:

- **Mac:** `brew install git` or it will prompt you to install Xcode command-line tools
- **Windows:** Download from [git-scm.com](https://git-scm.com/downloads)
- **Linux:** `sudo apt install git`

---

## Step 3: Create a Virtual Environment

A virtual environment keeps this project's packages separate from the rest of your system.

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

You'll see `(venv)` at the start of your terminal prompt when it's active.

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5: Create a Webex Bot

1. Go to [developer.webex.com](https://developer.webex.com/) and sign in
2. Click your profile icon → **My Webex Apps** → **Create a New App** → **Create a Bot**
3. Fill in:
   - **Bot Name:** Whatever you want (e.g., "My AI Bot")
   - **Bot Username:** Must be unique (e.g., "yourname-ai-bot")
   - **Icon:** Pick any
   - **Description:** Anything
4. Click **Add Bot**
5. **Copy the Bot Access Token** — you'll need it in the next step

> You can always get back to your token at [developer.webex.com/my-apps](https://developer.webex.com/my-apps)

---

## Step 6: Configure Environment Variables

```bash
cp .env.sample .env
```

Open `.env` in any text editor and fill in:

### Option A: OpenAI

```
BOT_TOKEN=paste-your-bot-token-here
AI_PROVIDER=openai
AI_API_KEY=sk-your-openai-key
```

Get an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Option B: Anthropic Claude

```
BOT_TOKEN=paste-your-bot-token-here
AI_PROVIDER=claude
AI_API_KEY=sk-ant-your-key
```

Get an API key at [console.anthropic.com](https://console.anthropic.com/)

### Option C: AWS Bedrock

```
BOT_TOKEN=paste-your-bot-token-here
AI_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
```

Requires an AWS account with Bedrock model access enabled for Claude.

---

## Step 7: Run the Bot

```bash
python bot.py
```

You should see:

```
=== Webex AI Bot ===
AI Provider: openai (gpt-4o)
Room restriction: disabled
Bot is running...
```

---

## Step 8: Talk to Your Bot

1. Open [web.webex.com](https://web.webex.com/)
2. Search for your bot's username (the one you created in Step 5)
3. Send it a message — it should respond with the TARS personality

---

## Stopping the Bot

Press `Ctrl+C` in the terminal to stop.

## Running Again Later

Every time you come back, activate the virtual environment first:

```bash
# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Then run:

```bash
python bot.py
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python: command not found` | Try `python3` instead, or check your PATH |
| `ERROR: BOT_TOKEN is missing` | Make sure you edited `.env` and pasted your token |
| Bot runs but doesn't respond | Make sure you're sending a **direct message** to the bot |
| `ModuleNotFoundError` | Make sure your virtual environment is activated (`source venv/bin/activate`) |
| Slow first response | Normal — the AI needs to "warm up". Subsequent messages will be faster |

---

## Next Steps

- Change `SYSTEM_PROMPT` in `bot.py` to customize the personality
- Try different AI providers and compare their responses
- Add the bot to a group space and use room restriction (`ALLOWED_ROOMS`)
- Check out [claude-webex-bridge](https://github.com/bryantbarzola/claude-webex-bridge) for an agentic bot that can execute tasks
