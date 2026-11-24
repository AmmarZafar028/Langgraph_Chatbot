from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import traceback

load_dotenv()

# -------------------
# Dedicated async loop for backend tasks
# -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


def _unwrap_exception(exc, indent=0):
    """Recursively unwrap ExceptionGroup to find root causes."""
    prefix = "  " * indent
    if hasattr(exc, "exceptions"):
        print(f"{prefix}[ExceptionGroup] {exc}")
        for sub in exc.exceptions:
            _unwrap_exception(sub, indent + 1)
    else:
        print(f"{prefix}[Root Cause] {type(exc).__name__}: {exc}")


# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI()

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


# -------------------
# MCP Client
# FIX: Each server is loaded independently so one failure doesn't block the other
# -------------------
def _make_client(config: dict):
    return MultiServerMCPClient(config)


async def _load_single_server(name: str, config: dict) -> list[BaseTool]:
    """Load tools from a single MCP server, returning [] on failure."""
    try:
        c = _make_client({name: config})
        tools_list = await c.get_tools()
        print(f"[MCP] '{name}' loaded {len(tools_list)} tool(s).")
        return tools_list
    except Exception as e:
        print(f"[MCP] '{name}' failed to load:")
        _unwrap_exception(e, indent=1)
        return []


async def _load_all_mcp_tools() -> list[BaseTool]:
    # FIX: arith uses the correct Windows python command
    # ⚠️  UPDATE the arith path below to your actual mcp-math-server/main.py location
    servers = {
        "arith": {
            "transport": "stdio",
            "command": "python",
            "args": ["F:/Langgraph_Chatbot/mcp-math-server/main.py"],  # <-- UPDATE THIS
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }

    all_tools = []
    for name, config in servers.items():
        server_tools = await _load_single_server(name, config)
        all_tools.extend(server_tools)
    return all_tools


def load_mcp_tools() -> list[BaseTool]:
    return run_async(_load_all_mcp_tools())


mcp_tools = load_mcp_tools()
tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm
print(f"[Tools] Active tools: {[t.name for t in tools]}")

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
print("[Graph] Chatbot graph compiled successfully.")

# -------------------
# 7. Helpers
# -------------------
async def _alist_threads():
    all_threads = set()
    try:
        async for checkpoint in checkpointer.alist(None):
            tid = checkpoint.config.get("configurable", {}).get("thread_id")
            if tid:
                all_threads.add(tid)
    except Exception as e:
        print(f"[Checkpointer] Could not list threads: {e}")
    return list(all_threads)


def retrieve_all_threads() -> list[str]:
    """Return all thread IDs stored in the checkpoint DB."""
    return run_async(_alist_threads())


async def chat(message: str, thread_id: str = "default") -> str:
    """Send a message and get a response for a given thread."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    return result["messages"][-1].content


# -------------------
# 8. Entry point — use streamlit, not this directly
#    Run: streamlit run streamlit_frontend_mcp.py
# -------------------
if __name__ == "__main__":
    print("\n✅ Backend loaded successfully.")
    print("   Run the app with:  streamlit run streamlit_frontend_mcp.py")
    print(f"   Active tools: {[t.name for t in tools]}")