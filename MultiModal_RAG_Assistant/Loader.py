from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, TextLoader
from langchain.docstore.document import Document
from PIL import Image, UnidentifiedImageError
from unstructured.partition.utils.ocr_models.tesseract_ocr import OCRAgentTesseract
import os

# Set paths for unstructured PDF and Tesseract
os.environ["PATH"] = os.path.abspath(r"Z:\Genai_Projects\poppler-25.07.0\Library\bin") + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = os.path.abspath(r"Z:\Genai_Projects\teseract") + os.pathsep + os.environ["PATH"]

class Loader:
    def __init__(self, folder_path: str ="MultiModal_RAG_Assistant/Knowledge_Base"):
        self.folder_path = folder_path
        self.docs = self.load_documents()

    def load_documents(self):
        all_docs = []

        try:
            ocr_agent = OCRAgentTesseract()
            # Load PDFs
            pdf_loader = DirectoryLoader(
                self.folder_path,
                glob="*.pdf",
                loader_cls=UnstructuredPDFLoader,
                loader_kwargs={"ocr_agent": ocr_agent}
            )
            pdf_docs = pdf_loader.load()
            print(f"✅ Loaded {len(pdf_docs)} PDF documents")
            all_docs.extend(pdf_docs)

        except Exception as e:
            print(f"⚠️ Error loading PDFs: {e}")

        try:
            # Load text files
            text_loader = DirectoryLoader(
                self.folder_path,
                glob="*.txt",
                loader_cls=TextLoader
            )
            text_docs = text_loader.load()
            print(f"✅ Loaded {len(text_docs)} text documents")
            all_docs.extend(text_docs)

        except Exception as e:
            print(f"⚠️ Error loading text files: {e}")

        try:
            # Load images
            image_docs = []
            for f in os.listdir(self.folder_path):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(self.folder_path, f)
                    try:
                        img = Image.open(file_path)
                        image_docs.append(
                            Document(
                                page_content=str(img),  # store metadata instead of raw PIL object
                                metadata={"source": f, "type": "image"}
                            )
                        )
                    except UnidentifiedImageError:
                        print(f"⚠️ Could not identify image: {f}")
                    except Exception as e:
                        print(f"⚠️ Error loading image {f}: {e}")

            print(f"✅ Loaded {len(image_docs)} image documents")
            all_docs.extend(image_docs)

        except Exception as e:
            print(f"⚠️ Error scanning images: {e}")

        if not all_docs:
            print("⚠️ No documents were loaded from the folder.")
        else:
            print(f"📂 Total documents loaded: {len(all_docs)}")

        return all_docs

    def get_images(self):
        """
        Return only the image documents from the loaded documents.
        """
        return [doc for doc in self.docs if doc.metadata.get("type") == "image"]

