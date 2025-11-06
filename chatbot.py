from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
# import os
# load_dotenv() 
# api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
class GraphState(TypedDict):
    question: str
    answer: str

def respond(state:GraphState):
    question = state['question']
    response = model.invoke(question)
    return {"answer":response.content}

graph = StateGraph(GraphState)
graph.add_node("AnswerNode", respond)
graph.set_entry_point("AnswerNode")
graph.add_edge("AnswerNode", END)
app = graph.compile()