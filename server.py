import json
import os

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("My MCP Server")

API_BASE_URL = os.getenv("API_BASE_URL", "")

# In-memory token store for the session
_auth_token: str = ""


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
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"username": username, "password": password},
            timeout=30.0,
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

    async with httpx.AsyncClient() as client:
        json_body = None
        if body and method.upper() in ("POST", "PUT", "PATCH"):
            json_body = json.loads(body)

        response = await client.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=json_body,
            timeout=30.0,
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

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=30.0)

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

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=30.0)

    if response.status_code in (200, 201):
        return f"Spend created successfully.\n{response.text}"
    else:
        return f"Error: Status {response.status_code}\n{response.text}"


if __name__ == "__main__":
    mcp.run()
