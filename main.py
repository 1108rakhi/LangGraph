from fastapi import FastAPI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
import uuid

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

class State(MessagesState):
    pass

app = FastAPI()
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

memory = MemorySaver()
graph = StateGraph(State)

def merge_messages(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return left + right

class MyGraph:
    def __init__(self):
        self.messages = []

    def add_message(self, new_msgs):
        self.messages = merge_messages(self.messages, new_msgs)

def chat_node(state: State):
    response = model.invoke(state["messages"])
    return {"messages": [("assistant", response.content)]}

graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chat_app = graph.compile(checkpointer=memory)

@app.post("/chat")
def chat_api(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    result = chat_app.invoke(
        {"messages": [("user", request.question)]},
        config={"configurable": {"thread_id": session_id}},
    )

    reply = result["messages"][-1].content
    return {"reply": reply, "session_id": session_id}
