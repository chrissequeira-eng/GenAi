from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel


class MultimodalWrapper:
    def __init__(self, env_path="myenv/.env",
                 llm_model="gemini-1.5-flash",
                 clip_model="openai/clip-vit-base-patch32"):
        """
        Wrapper for Gemini (LLM) and CLIP (embedder).
        """
        # Load environment variables
        load_dotenv(env_path)

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0
        )

        # Initialize CLIP model + processor
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)

