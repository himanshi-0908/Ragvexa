from langchain_nvidia_ai_endpoints import ChatNVIDIA
from config import NVIDIA_API_KEY

def get_llm():
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY is not set in .env")
    return ChatNVIDIA(
        nvidia_api_key=NVIDIA_API_KEY,
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.2
    )

def generate_response(llm, context, question, history_string=""):
    prompt = f"""Use the following pieces of context and previous conversation history to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Previous Conversation History:
{history_string}

Context: 
{context}

Question: {question}
Answer:"""
    response = llm.invoke(prompt)
    return response.content

def generate_study_guide(llm, context):
    prompt = f"""Based on the provided context from the student's study materials, generate a comprehensive study guide.
Include:
1. Key Concepts (bulleted list)
2. Important Definitions
3. 5 Flashcard-style Questions and Answers for review.

Context:
{context}

Study Guide:"""
    response = llm.invoke(prompt)
    return response.content
