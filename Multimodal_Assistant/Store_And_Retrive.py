from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from typing import List
import torch
from transformers import CLIPProcessor, CLIPModel
from Loader import Loader
from Text_splitter import split_documents
from PIL import Image
import os


# ----------- Custom CLIP Embeddings Wrapper -----------
class CLIPEmbeddings(Embeddings):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_documents(self, docs):
        embeddings = []
        for doc in docs:
            # Determine if doc is a Document or string
            if isinstance(doc, Document):
                content = doc.page_content
                source = doc.metadata.get("source", "")
            else:
                content = doc
                source = ""

            if source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
                image = Image.open(source).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    emb = self.model.get_image_features(**inputs)
                embeddings.append(emb.cpu().numpy()[0].tolist())
            else:
                inputs = self.processor(
                    text=[content], return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                with torch.no_grad():
                    emb = self.model.get_text_features(**inputs)
                embeddings.append(emb.cpu().numpy()[0].tolist())

        return embeddings


    def embed_query(self, query: str):
        """
        Embed a query for retrieval. Handles both text and image paths.
        """
        if os.path.exists(query) and query.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
        ):
            # Handle image input
            image = Image.open(query).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
            return embeddings.cpu().numpy()[0].tolist()
        else:
            # Handle text input
            inputs = self.processor(
                text=[query], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
            return embeddings.cpu().numpy()[0].tolist()


# ----------- Vector Store Handler -----------
class CLIPVectorStoreHandler:
    def __init__(self, persist_directory: str = "chroma_db", model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the vector store handler with CLIP embeddings.
        """
        self.persist_directory = persist_directory
        self.embedder = CLIPEmbeddings(model_name=model_name)  # embedder lives here
        self.vectorstore = None

    def store_documents(self, documents: List[Document]):
        """
        Store documents in Chroma using CLIP embeddings.
        """
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            persist_directory=self.persist_directory,
        )
        print(f"Stored {len(documents)} documents in vector store at '{self.persist_directory}'.")

    def load_vectorstore(self):
        """
        Load persisted Chroma vectorstore.
        """
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedder,
            )
            print(f"Loaded vector store from '{self.persist_directory}'.")

    def get_retriever(self, k: int = 5):
        """
        Get retriever interface.
        """
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.as_retriever(type="similarity",search_kwargs={"k": k})
    
    def unload_vectorstore(self):
        """Gracefully unload / close the vectorstore."""
        if self.vectorstore:
            try:
                self.vectorstore._client.close()   # Close DB connection (Chroma)
            except Exception as e:
                print(f"Warning: could not fully close vectorstore: {e}")
            finally:
                self.vectorstore = None
                print("Vectorstore unloaded successfully.")


# ---------------- TESTING ----------------
if __name__ == "__main__":
    folder = r"Z:\Genai_Projects\Multimodal_Assistant\Knowledge_Base"
    loader = Loader()
    documents = loader.load_directory(folder)
    split_docs = split_documents(documents, chunk_size=1000, chunk_overlap=100)
    handler = CLIPVectorStoreHandler(persist_directory="chroma_test_db")
    handler.store_documents(split_docs)

    # Store docs

    handler.load_vectorstore()

    # Query with text
    retriever = handler.get_retriever(k=2)

    text_query = "who is chris"
    results = retriever.invoke(text_query)
    print("\nRetriever Results for text query:")
    for r in results:
        print(f"- {r.metadata['source']}: {r.page_content}")

    # Query with image
    image_query = "what are the types of retrievers"
    if os.path.exists(image_query):
        results = retriever.invoke(image_query)
        print("\nRetriever Results for image query:")
        for r in results:
            print(f"- {r.metadata['source']}: {r.page_content}")
