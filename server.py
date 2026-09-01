import json
import os

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP(
    "My MCP Server",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8888")),
)

API_BASE_URL = os.getenv("API_BASE_URL", "")

# Shared HTTP client — reuses TCP connections across tool calls
_http_client = httpx.AsyncClient(timeout=30.0)

_TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".session_token")


def _load_saved_token() -> str:
    try:
        with open(_TOKEN_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _save_token(token: str) -> None:
    with open(_TOKEN_FILE, "w") as f:
        f.write(token)


def _clear_token() -> None:
    try:
        os.remove(_TOKEN_FILE)
    except FileNotFoundError:
        pass


# Load saved token from previous session
_auth_token: str = _load_saved_token()


@mcp.tool()
def check_session() -> str:
    """Check if there is an active saved session (logged in or not)."""
    if _auth_token:
        return "Session active. You are already logged in."
    return "No active session. Please login first."


@mcp.tool()
def logout() -> str:
    """Logout and clear the saved session token."""
    global _auth_token
    _auth_token = ""
    _clear_token()
    return "Logged out. Session cleared."


@mcp.tool()
def hello(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool()
async def login(username: str, password: str) -> str:
    """Login to the API with username and password to get a JWT token.
    Must be called before making any api_request.

    Args:
        username: User username.
        password: User password.
    """
    global _auth_token

    if not API_BASE_URL:
        return "Error: API_BASE_URL is not set in .env"

    url = f"{API_BASE_URL.rstrip('/')}/api/token/"
    response = await _http_client.post(
        url,
        json={"username": username, "password": password},
    )

    if response.status_code == 200:
        data = response.json()
        _auth_token = (
            data.get("access")
            or data.get("access_token")
            or data.get("token")
            or ""
        )
        if not _auth_token:
            return f"Login response missing token. Keys: {list(data.keys())}"
        _save_token(_auth_token)
        return "Login successful. You can now make API requests."
    else:
        return f"Login failed. Status: {response.status_code}\n{response.text}"


@mcp.tool()
async def api_request(endpoint: str, method: str = "GET", body: str = "") -> str:
    """Make an authenticated request to the custom API.
    User must login first.

    Args:
        endpoint: The API endpoint path (e.g. '/users', '/data').
        method: HTTP method - GET, POST, PUT, DELETE.
        body: JSON string for request body (for POST/PUT).
    """
    if not API_BASE_URL:
        return "Error: API_BASE_URL is not set in .env"

    if not _auth_token:
        return "Error: Not authenticated. Please call the login tool first."

    url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {_auth_token}",
    }

    json_body = None
    if body and method.upper() in ("POST", "PUT", "PATCH"):
        json_body = json.loads(body)

    response = await _http_client.request(
        method=method.upper(),
        url=url,
        headers=headers,
        json=json_body,
    )
    return f"Status: {response.status_code}\n{response.text}"


@mcp.tool()
async def get_spends_by_month(month_id: int) -> str:
    """Retrieve a list of spends for a given month.

    Args:
        month_id: Month ID as an integer (1 = January, 12 = December).
    """
    if not API_BASE_URL:
        return "Error: API_BASE_URL is not set in .env"

    if not _auth_token:
        return "Error: Not authenticated. Please call the login tool first."

    url = f"{API_BASE_URL.rstrip('/')}/api/spends/month/{month_id}/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {_auth_token}",
    }

    response = await _http_client.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        return f"Error: Status {response.status_code}\n{response.text}"


@mcp.tool()
async def create_spend(
    date: str,
    description: str,
    amount: float,
    spend_type: int,
    spend_rate: int,
    location: str,
) -> str:
    """Create a new spend entry.

    Args:
        date: Date of the spend in YYYY-MM-DD format.
        description: Description of the spend.
        amount: Amount spent.
        spend_type: Spend type ID.
        spend_rate: Spend rate ID.
        location: Location of the spend.
    """
    if not API_BASE_URL:
        return "Error: API_BASE_URL is not set in .env"

    if not _auth_token:
        return "Error: Not authenticated. Please call the login tool first."

    url = f"{API_BASE_URL.rstrip('/')}/api/spends/new/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {_auth_token}",
    }
    payload = {
        "date": date,
        "description": description,
        "amount": amount,
        "spend_type": spend_type,
        "spend_rate": spend_rate,
        "location": location,
    }

    response = await _http_client.post(url, headers=headers, json=payload)

    if response.status_code in (200, 201):
        return f"Spend created successfully.\n{response.text}"
    else:
        return f"Error: Status {response.status_code}\n{response.text}"


@mcp.tool()
async def search_spends_by_category(month_id: int, spend_type: int) -> str:
    """Search spends for a given month filtered by category (spend type).

    Args:
        month_id: Month ID as an integer (1 = January, 12 = December).
        spend_type: Spend type ID to filter by. Use get_spend_types to see available types.
    """
    if not API_BASE_URL:
        return "Error: API_BASE_URL is not set in .env"
    if not _auth_token:
        return "Error: Not authenticated. Please call the login tool first."

    url = f"{API_BASE_URL.rstrip('/')}/api/spends/month/{month_id}/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {_auth_token}",
    }

    response = await _http_client.get(url, headers=headers)

    if response.status_code != 200:
        return f"Error: Status {response.status_code}\n{response.text}"

    spends = json.loads(response.text)
    filtered = [s for s in spends if s.get("spend_type") == spend_type]
    total = sum(float(s.get("amount", 0)) for s in filtered)
    return json.dumps({
        "spend_type": spend_type,
        "month": month_id,
        "count": len(filtered),
        "total": total,
        "spends": filtered,
    }, indent=2)


@mcp.tool()
async def get_spends_grouped_by_location(month_id: int) -> str:
    """Get spends for a month grouped by location with totals per location.

    Args:
        month_id: Month ID as an integer (1 = January, 12 = December).
    """
    if not API_BASE_URL:
        return "Error: API_BASE_URL is not set in .env"
    if not _auth_token:
        return "Error: Not authenticated. Please call the login tool first."

    url = f"{API_BASE_URL.rstrip('/')}/api/spends/month/{month_id}/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {_auth_token}",
    }

    response = await _http_client.get(url, headers=headers)

    if response.status_code != 200:
        return f"Error: Status {response.status_code}\n{response.text}"

    spends = json.loads(response.text)
    groups: dict[str, list] = {}
    for s in spends:
        loc = s.get("location", "Unknown") or "Unknown"
        groups.setdefault(loc, []).append(s)

    summary = {}
    for loc, items in groups.items():
        total = sum(float(i.get("amount", 0)) for i in items)
        summary[loc] = {"count": len(items), "total": round(total, 2)}

    return json.dumps({"month": month_id, "locations": summary}, indent=2)


@mcp.tool()
async def summarize_spends(month_id: int) -> str:
    """Generate a full summary of spends for a month: total spent, breakdown
    by category (spend type) and by location, plus the top 5 largest spends.

    Args:
        month_id: Month ID as an integer (1 = January, 12 = December).
    """
    if not API_BASE_URL:
        return "Error: API_BASE_URL is not set in .env"
    if not _auth_token:
        return "Error: Not authenticated. Please call the login tool first."

    url = f"{API_BASE_URL.rstrip('/')}/api/spends/month/{month_id}/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {_auth_token}",
    }

    response = await _http_client.get(url, headers=headers)

    if response.status_code != 200:
        return f"Error: Status {response.status_code}\n{response.text}"

    spends = json.loads(response.text)
    if not spends:
        return json.dumps({"month": month_id, "message": "No spends found."})

    grand_total = sum(float(s.get("amount", 0)) for s in spends)

    by_type: dict[str, dict] = {}
    for s in spends:
        st = str(s.get("spend_type", "Unknown"))
        entry = by_type.setdefault(st, {"count": 0, "total": 0.0})
        entry["count"] += 1
        entry["total"] += float(s.get("amount", 0))

    by_location: dict[str, dict] = {}
    for s in spends:
        loc = s.get("location", "Unknown") or "Unknown"
        entry = by_location.setdefault(loc, {"count": 0, "total": 0.0})
        entry["count"] += 1
        entry["total"] += float(s.get("amount", 0))

    top_spends = sorted(spends, key=lambda x: float(x.get("amount", 0)), reverse=True)[:5]

    return json.dumps({
        "month": month_id,
        "total_spends": len(spends),
        "grand_total": grand_total,
        "by_category": by_type,
        "by_location": by_location,
        "top_5_spends": top_spends,
    }, indent=2)


@mcp.tool()
async def get_spend_types() -> str:
    """Retrieve the list of available spend types."""
    if not API_BASE_URL:
        return "Error: API_BASE_URL is not set in .env"

    if not _auth_token:
        return "Error: Not authenticated. Please call the login tool first."

    url = f"{API_BASE_URL.rstrip('/')}/api/spend-types/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {_auth_token}",
    }

    response = await _http_client.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        return f"Error: Status {response.status_code}\n{response.text}"


if __name__ == "__main__":
    import sys as _sys

    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        print(
            f"Starting MCP SSE server on {mcp.settings.host}:{mcp.settings.port}",
            file=_sys.stderr,
        )
        mcp.run(transport="sse")
    else:
        mcp.run()
