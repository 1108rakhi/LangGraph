# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import app as langgraph_app

app = FastAPI(title="LangGraph")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    result = langgraph_app.invoke({"question": req.question})
    return {"answer": result["answer"]}
