from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize LLM properly
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ✅ Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ✅ Define chat node correctly
def chat_node(state):
    messages = state["messages"]  # ✅ extract messages from state
    response = llm.invoke(messages)  # ✅ pass messages to LLM
    return {"messages": [response]}  # ✅ return correctly

# ✅ Build Graph
graph = StateGraph(State)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

# ✅ Compile with memory
memory = MemorySaver()
chatbot = graph.compile(checkpointer=memory)






    