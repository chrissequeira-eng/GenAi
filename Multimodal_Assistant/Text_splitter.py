from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Split only text-based documents. Image docs are passed through unchanged.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    split_docs: List[Document] = []
    
    for doc in documents:
        doc_type = doc.metadata.get("type", "text")  # default to text
        
        if doc_type == "image":
            # Keep image docs as they are (no splitting)
            split_docs.append(doc)
        else:
            # Split text/pdf/ocr docs
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunk": i}
                    )
                )
    
    return split_docs

# Example usage
if __name__ == "__main__":
    from Loader import Loader  # Assuming your Loader class is in Loader.py

    folder = r"Z:\Genai_Projects\Multimodal_Assistant\Knowledge_Base"
    loader = Loader()
    documents = loader.load_directory(folder)

    print(f"Original documents: {len(documents)}")

    split_docs = split_documents(documents, chunk_size=1000, chunk_overlap=100)
    print(f"Total chunks after splitting: {len(split_docs)}\n")

    # Preview first 5 chunks
    for i, doc in enumerate(split_docs[:5]):
        print(f"Chunk {i+1}:")
        print("Metadata:", doc.metadata)
        print("Content preview:", doc.page_content[:200].replace("\n", " "), "\n---\n")
