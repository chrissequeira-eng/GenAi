from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from PIL import Image
import pytesseract
import os

class TextSplitter:
    def __init__(self, chunk_size=200, chunk_overlap=20):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str):
        docs = [Document(page_content=text)]
        split_docs = self.splitter.split_documents(docs)
        return [doc.page_content for doc in split_docs]

    def split_file(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        extracted_text = ""

        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()

        elif ext == ".pdf":
            loader = UnstructuredPDFLoader(file_path)
            docs = loader.load()
            extracted_text = "\n".join([doc.page_content for doc in docs])

        elif ext in [".png", ".jpg", ".jpeg"]:
            img = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(img)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return self.split_text(extracted_text)

    def split_loader_docs(self, loader_docs):
        all_chunks = []
        for doc in loader_docs:
            # If it's a LangChain Document
            if hasattr(doc, "page_content"):
                all_chunks.extend(self.split_text(doc.page_content))

            # If it's a file path
            elif isinstance(doc, str) and os.path.exists(doc):
                all_chunks.extend(self.split_file(doc))

            # If it's raw text string
            elif isinstance(doc, str):
                all_chunks.extend(self.split_text(doc))

            else:
                raise ValueError(f"Unsupported Loader output type: {type(doc)}")

        return all_chunks