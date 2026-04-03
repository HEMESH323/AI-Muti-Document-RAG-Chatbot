import google.generativeai as genai
from src.utils import logger


class ChatbotManager:
    """Manages the Chatbot logic using Google Gemini RAG."""

    def __init__(self, retriever, memory):
        self.model = genai.GenerativeModel("models/gemini-flash-latest")
        self.retriever = retriever
        self.memory = memory

        logger.info("✅ ChatbotManager initialized with Gemini")

    def ask(self, query: str):
        try:
            # Handle missing retriever (for safety)
            if self.retriever:
                docs = self.retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in docs])
            else:
                docs = []
                context = "No context available."

            prompt = f"""
You are an intelligent assistant. Answer strictly using the provided context.
If the answer is not found, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

            response = self.model.generate_content(prompt)

            # Safe response handling
            if not response or not response.text:
                return "No response from model.", docs

            return response.text, docs

        except Exception as e:
            logger.error(f"Error in ask(): {str(e)}")
            return f"Error: {str(e)}", []
    
