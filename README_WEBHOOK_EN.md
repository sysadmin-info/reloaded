# Drone Navigation Webhook

Solution to the "webhook" task (Task 18) - an API that interprets drone flight instructions on a 4x4 map using LangGraph orchestration.

## Task Description

1. **4x4 Map** - the drone moves on a grid:

   ```
   [1,1] Start    [1,2] Meadow     [1,3] Tree      [1,4] House
   [2,1] Meadow   [2,2] Windmill   [2,3] Meadow    [2,4] Meadow
   [3,1] Meadow   [3,2] Meadow     [3,3] Rocks     [3,4] Trees
   [4,1] Mountains [4,2] Mountains [4,3] Car       [4,4] Cave
   ```

2. **API** accepts natural language instructions and returns what is found at the final position

3. **The drone always starts** from position \[1,1] (top-left corner)

## Quick Start

### 1. Prepare the Environment

```bash
# Edit .env and set API keys
nano .env

# Install dependencies
pip install fastapi uvicorn python-dotenv langgraph langchain-core requests pydantic openai

# For other engines:
pip install anthropic  # for Claude
pip install google-generativeai  # for Gemini
pip install langchain-anthropic  # for Claude with agent.py
```

### 2. Install ngrok

```bash
# macOS
brew install ngrok

# Ubuntu
snap install ngrok

# Or download from https://ngrok.com/download

# WSL / Linux without snap
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok

# TOKEN - set up an account and generate token here: https://dashboard.ngrok.com/signup
ngrok config add-authtoken YOUR_TOKEN
```

### 3. Run the Webhook

#### Option A: Direct Run with Multi-Engine Support (recommended)

```bash
# OpenAI (default)
python zad18.py --engine openai

# Claude 
python zad18.py --engine claude

# Gemini
python zad18.py --engine gemini

# Local models (LMStudio)
python zad18.py --engine lmstudio

# Anything LLM
python zad18.py --engine anything
```

#### Option B: Through Agent (for clean flag extraction)

```bash
# Start the agent
python agent.py

# Choose your engine when prompted, then run:
> run_task 18

# Agent will extract only the flag: {{FLG:}}
```

#### Option C: Custom Options

```bash
# Use different port
python zad18.py --engine claude --port 3002

# Don't send URL automatically (manual testing)
python zad18.py --engine openai --skip-send
```

### 4. Local Testing

```bash
# Test a single instruction while webhook is running
curl -X POST http://localhost:3001/ \
  -H "Content-Type: application/json" \
  -d '{"instruction":"go right and down"}'

# Expected response:
# {"description":"windmill","_thinking":"..."}
```

## Multi-Engine Architecture

The webhook automatically detects and uses the specified LLM engine:

```python
# Supported engines:
--engine openai     # GPT models via OpenAI API
--engine claude     # Claude models via Anthropic API  
--engine gemini     # Gemini models via Google API
--engine lmstudio   # Local models via LMStudio
--engine anything   # Local models via Anything LLM
```

### LangGraph Pipeline

```python
[START]
   ↓
[check_environment] - Verify ngrok and port availability
   ↓
[start_server] - Launch FastAPI webhook in background
   ↓
[start_ngrok] - Create public tunnel and get URL
   ↓
[send_webhook_url] - Send URL to central server
   ↓
[wait_for_completion] - Wait for tests (or auto-exit on flag)
   ↓
[cleanup] - Stop all processes
   ↓
[END]
```

### Drone Navigation with LangGraph

```python
[START]
   ↓
[parse_instruction] - LLM interprets the instruction into movements
   ↓
[execute_movements] - Simulates drone movement on the map
   ↓
[END] → returns the place description
```

### Key Components:

1. **WebhookState** - state of the webhook pipeline in LangGraph
2. **NavigationState** - state of drone navigation flow
3. **parse\_instruction\_node** - uses LLM to interpret instructions
4. **execute\_movements\_node** - executes movements and checks position
5. **FastAPI endpoints** - `/` for instructions, `/health` for health check

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Central server
CENTRALA_API_KEY=your-key
REPORT_URL=https://xxx.xxx.xxx/report

# LLM Engine (optional - can be set via --engine)
LLM_ENGINE=openai  # or: claude, lmstudio, gemini, anything

# API keys (depending on engine)
OPENAI_API_KEY=sk-...
CLAUDE_API_KEY=sk-ant-...
ANTHROPIC_API_KEY=sk-ant-...  # alternative to CLAUDE_API_KEY
GEMINI_API_KEY=AI...

# Models for each engine
MODEL_NAME_OPENAI=gpt-4o-mini
MODEL_NAME_CLAUDE=claude-sonnet-4-20250514
MODEL_NAME_GEMINI=gemini-2.5-pro-latest
MODEL_NAME_LM=llama-3.3-70b-instruct
MODEL_NAME_ANY=llama-3.3-70b-instruct

# Local model URLs
LMSTUDIO_API_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=local
ANYTHING_API_URL=http://localhost:1234/v1
ANYTHING_API_KEY=local
```

## Command Line Options

```bash
python zad18.py [OPTIONS]

Options:
  --engine {openai,claude,gemini,lmstudio,anything}
                        LLM backend to use
  --port INTEGER        Port for webhook server (default: 3001)
  --skip-send          Don't send URL to central server automatically
  --help               Show this message and exit
```

## Instruction Examples

The webhook handles complex natural language instructions:

1. **Simple**: "go right", "fly down"
2. **Complex**: "all the way right, then as far down as possible"
3. **With cancellation**: "fly down, or wait! go right instead"
4. **Multi-step**: "right, then down, then right again"
5. **Polish examples**: "na maksa w prawo, a później ile wlezie w dół"

## Troubleshooting

### "Port already in use"

```bash
# Find the process
lsof -i :3001

# Kill the process
kill -9 <PID>
```

### "Ngrok not working"

* Make sure you have a ngrok account (free)
* Set up authtoken: `ngrok config add-authtoken <your-token>`
* Check if ngrok is running: `curl http://localhost:4040/api/tunnels`

### "LLM doesn't interpret correctly"

* Check the model in .env (GPT-4 works better than GPT-3.5)
* Verify API keys are set correctly
* For local models, ensure LMStudio/Anything is running

### "Engine not found"

* Install required packages:
  ```bash
  pip install anthropic  # for Claude
  pip install google-generativeai  # for Gemini
  pip install langchain-anthropic  # for agent.py with Claude
  ```

## Response Structure

The API always returns:

```json
{
    "description": "place_name",
    "_thinking": "optional_debug_logs"
}
```

Where `description` is up to 2 words describing the destination.

## Getting the Flag

### Method 1: Direct Run
```bash
python zad18.py --engine claude
# Webhook will automatically exit after receiving: {{FLG:}}
```

### Method 2: Through Agent (Clean Output)
```bash
python agent.py
> run_task 18
# Returns only: 🏁 Flaga znaleziona: {{FLG:}} - kończę zadanie.
```

## Process Flow

1. **Environment Check** - Verify ngrok installation and port availability
2. **Server Launch** - Start FastAPI webhook server in background thread
3. **Tunnel Creation** - Launch ngrok and get public HTTPS URL
4. **Registration** - Send webhook URL to central server
5. **Testing Phase** - Central server tests the webhook with various instructions
6. **Flag Reception** - If all tests pass, receive flag and auto-exit
7. **Cleanup** - Stop all processes and exit cleanly

The entire process is automated and will complete without user intervention once started.