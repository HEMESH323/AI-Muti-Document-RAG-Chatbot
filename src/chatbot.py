from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
from src.utils import logger, get_env_variable

class ChatbotManager:
    """Manages the Chatbot logic using Google Gemini RAG."""
    
    def __init__(self, retriever, memory):
        api_key = get_env_variable("GOOGLE_API_KEY")
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
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )

    def ask(self, query: str):
        """Asks a question and returns the answer with source documents."""
        logger.info(f"Asking chatbot: {query}")
        result = self.chain.invoke({"question": query})
        return result["answer"], result["source_documents"]

