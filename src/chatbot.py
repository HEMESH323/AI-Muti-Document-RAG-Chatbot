import google.generativeai as genai
import os
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from src.utils import logger, get_env_variable

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))    
class ChatbotManager:
    """Manages the Chatbot logic using Google Gemini RAG."""
    
    def __init__(self, retriever, memory):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.retriever = retriever
        self.memory = memory
        logger.info("Initializing ChatbotManager with Gemini...")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=api_key,
            temperature=0.3
        )
        
        self.memory = memory
        self.retriever = retriever
        
        # System prompt template
        self.template = """
        You are an intelligent assistant. Answer strictly using the provided context. 
        If the answer is not found in the context, say you don't know and do not try to make up an answer.
        
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        
        Answer:"""
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "chat_history", "question"]
        )
        


def ask(self, query: str):
    docs = self.retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an intelligent assistant. Answer strictly using the provided context.
    If the answer is not found, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """

    response = self.model.generate_content(prompt)

    return {
        "answer": response.text,
        "source_documents": docs
    }

