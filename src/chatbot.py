from google import genai
import os

class ChatbotManager:
    def __init__(self, retriever, memory):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.retriever = retriever
        self.memory = memory

    def ask(self, query: str):
        docs = self.retriever.invoke(query)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an intelligent assistant. Answer strictly using the provided context.
        If the answer is not found, say "I don't know".

        Context:
        {context}

        Question:
        {query}
        """

        response = self.client.models.generate_content(
            model="gemini-1.5-flash-8b",
            contents=prompt   # ✅ CORRECT
        )

        return response.text, docs
