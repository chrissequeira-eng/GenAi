from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from typing import List
import pytesseract
import os
import glob
import fitz  # PyMuPDF
from PIL import Image

# Set local Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\teseract\tesseract.exe"

class Loader:
    Text_ext = ["*.txt", "*.md"]
    PDF_ext = ["*.pdf"]
    Image_ext = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff"]

    # Custom PDF loader
    def load_pdf_as_document(self, file_path: str) -> Document:
        full_text = ""
        try:
            pdf = fitz.open(file_path)
            for i, page in enumerate(pdf):
                try:
                    full_text += page.get_text()
                except Exception as e:
                    print(f"⚠️ Failed to extract text from page {i} in {file_path}: {e}")
            pdf.close()
        except Exception as e:
            print(f"⚠️ Failed to load PDF {file_path}: {e}")
            full_text = ""

        return Document(page_content=full_text, metadata={"source": file_path, "type": "pdf"})

    # Custom image loader → returns BOTH OCR doc + raw image doc
    def load_image_as_documents(self, file_path: str) -> List[Document]:
        docs = []
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"⚠️ Failed to OCR image {file_path}: {e}")
            text = ""

        # 1. OCR text doc (may be empty, still useful for retrieval)
        docs.append(Document(
            page_content=text,
            metadata={"source": file_path, "type": "ocr"}
        ))

        # 2. Image-only doc (raw path, ensures CLIP will embed the image)
        docs.append(Document(
            page_content=file_path,
            metadata={"source": file_path, "type": "image"}
        ))

        return docs

    # Main directory loader
    def load_directory(self, folder_path: str) -> List[Document]:
        all_docs: List[Document] = []

        # Load images individually
        for ext in self.Image_ext:
            for img_path in glob.glob(os.path.join(folder_path, "**", ext), recursive=True):
                img_docs = self.load_image_as_documents(img_path)
                all_docs.extend(img_docs)

        # Load text files using DirectoryLoader
        for ext in self.Text_ext:
            docs = DirectoryLoader(folder_path, glob=f"**/{ext}", loader_cls=TextLoader).load()
            for d in docs:
                d.metadata["type"] = "text"
            all_docs.extend(docs)

        # Load PDFs using custom function
        for ext in self.PDF_ext:
            for pdf_file in glob.glob(os.path.join(folder_path, "**", ext), recursive=True):
                pdf_doc = self.load_pdf_as_document(pdf_file)
                all_docs.append(pdf_doc)

        return all_docs


# ---------------- Test ----------------
if __name__ == "__main__":
    folder = r"Z:\Genai_Projects\Multimodal_Assistant\Knowledge_Base"
    loader = Loader()
    documents = loader.load_directory(folder)

    print(f"Total documents loaded: {len(documents)}\n")

    # Preview first 5 documents
    for i, doc in enumerate(documents[:5]):
        print(f"Document {i+1}:")
        print("Metadata:", doc.metadata)
        print("Content preview:", doc.page_content[:300].replace("\n", " "), "\n---\n")
