from langchain.docstore.document import Document
from langchain_chroma import Chroma
from Multimodal_Embedding import MultimodalEmbedding
import gc

class ChromaDBHandler:
    def __init__(self, persist_directory="chroma_db_gemini"):
        self.persist_directory = persist_directory
        self.vectorstore = None  # this is your Chroma instance

    def store_in_chroma(self, texts):
        # Wrap texts into Document objects
        docs = [Document(page_content=t) for t in texts]
        embedder=MultimodalEmbedding()

        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            embedding_function=embedder,
            persist_directory=self.persist_directory
        )

        # Store documents
        self.vectorstore.add_documents(docs)
        print(f"✅ Stored {len(docs)} documents in {self.persist_directory}")

    def unload(self):
        if self.vectorstore is not None:
            del self.vectorstore
            self.vectorstore = None
            gc.collect()  # force garbage collection
            print("🗑️ Chroma vectorstore unloaded from memory.")

    def get_retriever(self, k=8, search_type="similarity"):
        # Load vectorstore if not already loaded
        embedder=MultimodalEmbedding()
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedder
            )

        # Create and return retriever
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )