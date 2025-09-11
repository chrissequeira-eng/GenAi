import os
from Loader import Loader
from Store_And_Retrieve import ChromaDBHandler
from LLM_Image_Summarizer import GeminiMultimodal
from Chain_Setup import Chain_Setup
from PromptTemplate import prompt
from langchain_core.output_parsers import StrOutputParser
from Text_Splitter import TextSplitter

UPLOAD_DIR = r"Z:\Genai_Projects\MultiModal_RAG_Assistant\Knowledge_Base"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("🤖 Multimodal RAG Chatbot")

# Initialize LLM
llm_model = GeminiMultimodal(
    model="gemini-1.5-flash",
    temperature=0.5,
    dotenv_path="myenv/.env"
)
llm = llm_model.llm  # LangChain LLM instance
parser = StrOutputParser()

# Initialize Loader
loader = Loader()
docs = loader.docs

# Initialize conditional chain only if documents are available
if not docs:
    print("❌ No documents loaded.")
    retriever = None
    conditional_chain = None
else:
    # Split and store in Chroma
    splitter = TextSplitter()
    texts = splitter.split_loader_docs(docs)
    chroma = ChromaDBHandler()
    chroma.store_in_chroma(texts)
    retriever = chroma.get_retriever()
    chain =Chain_Setup(retriever, prompt, llm, parser)

# Chat loop
chat_history = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Bye!")
        if retriever:
            chroma.unload()  # Clean up Chroma resources
        break

    chat_history.append(("user", user_input))

    if retriever and chain:
        response=chain.invoke(user_input)
        chat_history.append(("assistant", response))
        print(f"Assistant: {response}")
"""
# Function to add new files
def add_file(file_path):
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return

    dest_path = os.path.join(UPLOAD_DIR, os.path.basename(file_path))
    with open(file_path, "rb") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())
    print(f"✅ Saved file to: {dest_path}")

    # Reload and process new docs
    loader.docs = Loader().docs  # reload using Loader class
    new_docs = loader.docs

    if new_docs:
        splitter = TextSplitter()
        new_texts = splitter.split_loader_docs(new_docs)
        chroma.store_in_chroma(new_texts)  # store in Chroma
        retriever = chroma.get_retriever()
        # Update the conditional chain
        global conditional_chain
        conditional_chain = Chain_Setup(retriever, prompt, llm, parser)
        print("📂 Knowledge base updated with new documents!")

"""
