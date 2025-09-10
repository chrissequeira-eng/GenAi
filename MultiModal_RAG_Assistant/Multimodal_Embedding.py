from langchain_core.embeddings import Embeddings
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np


class MultimodalEmbedding(Embeddings):
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def embed_query(self, input_data):
        """
        Embed a single query (either text or image path).
        """
        if isinstance(input_data, str) and input_data.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(input_data).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            return outputs.cpu().numpy()[0]
        else:
            inputs = self.processor(text=[input_data], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            return outputs.cpu().numpy()[0]

    def embed_documents(self, docs):
        """
        Embed multiple documents (list of text or image paths).
        """
        vectors = []
        for doc in docs:
            vectors.append(self.embed_query(doc))
        return vectors
