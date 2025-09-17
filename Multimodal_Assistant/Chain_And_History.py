from Loader import Loader
from Text_splitter import split_documents
from Store_And_Retrive import CLIPVectorStoreHandler
from Multimodal_Assistant.Main import prompt
from Setup import MultimodalWrapper

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ------------------- PIPELINE -------------------

def build_rag_chain(kb_path: str, persist_dir: str = "chroma_test_db"):

   # 5. Build ConversationalRetrievalChain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},  # custom instructions
    )

    return rag_chain
