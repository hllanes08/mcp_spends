# Spends MCP Server

A Model Context Protocol (MCP) server that connects to a Spends API, allowing LLMs to authenticate, query, and create spend entries through tool calls.

## Tools

| Tool | Description |
|------|-------------|
| `hello` | Greet someone by name |
| `add` | Add two numbers together |
| `login` | Authenticate with username and password to get an API token |
| `api_request` | Make an authenticated request to any API endpoint |
| `get_spends_by_month` | Retrieve a list of spends for a given month (1-12) |
| `create_spend` | Create a new spend entry with date, description, amount, type, rate, and location |

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file:

```
API_BASE_URL=http://localhost:8000/
GEMINI_API_KEY=your_gemini_api_key      # only needed for client.py
OLLAMA_MODEL=qwen3.5:latest             # only needed for client_ollama.py
```

## Usage

### With Claude Code

```bash
claude mcp add spends-api -s user -- python /path/to/server.py
```

Then ask Claude directly: "login and get spends for month 5".

### With Ollama (interactive chatbot)

```bash
.venv/bin/python client_ollama.py
```

### With Gemini (single prompt)

```bash
.venv/bin/python client.py "get spends for January"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/token/` | POST | Login with `username` and `password` |
| `/api/spends/month/{month_id}/` | GET | Get spends by month (1-12) |
| `/api/spends/new/` | POST | Create a new spend |

### Create Spend JSON format

```json
{
  "date": "2026-05-27",
  "description": "Gemini",
  "amount": 10,
  "spend_type": 1,
  "spend_rate": 1,
  "location": "Google"
}
```
