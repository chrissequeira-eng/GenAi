# ---------------- Imports ----------------
import os
from io import BytesIO
import base64
from PIL import Image
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda

from Loader import Loader
from Text_splitter import split_documents
from Store_And_Retrive import CLIPVectorStoreHandler
from Setup import MultimodalWrapper

# ---------------- Prompt Template ----------------
template = """
You are a helpful AI assistant.

Use the following context and attached images to answer the question.
Include the conversation history if available.
If the context and images are not sufficient, say: "I don't know."

Conversation History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=template
)

# ---------------- Helper Functions ----------------
def build_context_from_docs(retrieved_docs: List[Document], max_context_chars: int = 3000) -> str:
    """Convert retrieved docs into a text context string."""
    context_texts = []
    for doc in retrieved_docs:
        ocr_text = doc.metadata.get("ocr_text", "").strip()
        if ocr_text:
            context_texts.append(ocr_text)
        if doc.page_content.strip() and doc.metadata.get("type") != "image":
            context_texts.append(doc.page_content.strip())
    combined_context = "\n\n".join(context_texts)
    if len(combined_context) > max_context_chars:
        combined_context = combined_context[-max_context_chars:]
    return combined_context

def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for LLM."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

# ---------------- Chat History Class ----------------
class ChatMemory:
    """Simple in-memory chat history for conversation."""
    def __init__(self):
        self.history: List[str] = []

    def add(self, user_query: str, assistant_answer: str):
        self.history.append(f"User: {user_query}\nAssistant: {assistant_answer}")

    def get_history(self, max_chars: int = 1000) -> str:
        full_history = "\n".join(self.history)
        if len(full_history) > max_chars:
            return full_history[-max_chars:]
        return full_history

# ---------------- Pipeline Helper Functions ----------------
def context_step(input_tuple, chat_memory: ChatMemory):
    """Build context text and collect images, append chat history."""
    docs, query = input_tuple
    ctx = build_context_from_docs(docs)

    images = []
    for d in docs:
        if d.metadata.get("type") == "image":
            path = d.page_content.strip()
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    images.append(img)
                except Exception as e:
                    print(f"⚠️ Failed to open {path}: {e}")
            else:
                print(f"⚠️ Image path not found: {path}")

    return {"context": ctx, "question": query, "images": images, "chat_history": chat_memory.get_history()}

def multimodal_message(payload):
    """Convert context + images into LLM-ready HumanMessage."""
    context = payload["context"]
    question = payload["question"]
    chat_history = payload.get("chat_history", "")
    images = payload["images"]

    message = HumanMessage(
        content=[{"type": "text", "text": prompt_template.format(
            context=context, question=question, chat_history=chat_history
        )}]
    )

    for img in images:
        try:
            img_b64 = pil_to_base64(img)
            message.content.append({"type": "image_url", "image_url": img_b64})
        except Exception as e:
            print(f"⚠️ Error encoding image: {e}")

    return [message]

# ---------------- Main RAG Pipeline ----------------
def get_rag_pipeline(folder_path: str, persist_dir: str = "chroma_test_db", k: int = 2, env_path: str = "myenv/.env"):
    """Initialize a RAG-with-images pipeline with chat history."""
    chat_memory = ChatMemory()

    # Load documents
    loader = Loader()
    documents = loader.load_directory(folder_path)
    for doc in documents:
        if doc.metadata.get("type") == "ocr":
            doc.metadata["ocr_text"] = doc.page_content.strip()
    split_docs = split_documents(documents, chunk_size=1000, chunk_overlap=100)

    # Vector store
    handler = CLIPVectorStoreHandler(persist_directory=persist_dir)
    handler.store_documents(split_docs)
    handler.load_vectorstore()
    retriever = handler.get_retriever(k=k)

    # LLM
    wrapper = MultimodalWrapper(env_path=env_path)
    llm = wrapper.llm

    # Retrieval step
    def retrieval_step(query):
        docs = retriever.get_relevant_documents(query)
        print("\n--- Retrieved Docs ---")
        for d in docs:
            print(f"Source: {d.metadata.get('source')} | Preview: {d.page_content[:120]}...")
        return (docs, query)

    # Full pipeline
    pipeline = (
        RunnableLambda(retrieval_step) |
        RunnableLambda(lambda tup: context_step(tup, chat_memory)) |
        RunnableLambda(multimodal_message) |
        llm
    )

    # Wrapper to update chat memory
    def wrapped_pipeline(query: str):
        result = pipeline.invoke(query)
        answer_text = result.content if hasattr(result, "content") else str(result)
        chat_memory.add(query, answer_text)
        return result

    return wrapped_pipeline

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    folder = r"Z:\Genai_Projects\Multimodal_Assistant\Knowledge_Base"
    rag_pipeline = get_rag_pipeline(folder_path=folder)
    handle=CLIPVectorStoreHandler()

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            handle.unload_vectorstore()
            print("exiting....")
            break
        answer = rag_pipeline(user_query)
        print("\nAssistant:", answer.content if hasattr(answer, "content") else answer)
