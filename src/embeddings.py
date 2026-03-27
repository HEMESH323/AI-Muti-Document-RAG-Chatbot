from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

    def get_embeddings(self):
        return self.embeddings
