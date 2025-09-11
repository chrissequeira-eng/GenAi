from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import base64
import os


class GeminiMultimodal:
    def __init__(self, model="gemini-1.5-flash", temperature=0, dotenv_path="myenv/.env"):
        load_dotenv(dotenv_path=dotenv_path)
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )

    def _encode_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # detect extension
        ext = os.path.splitext(image_path)[-1].lower()
        if ext == ".png":
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"
        
        return f"data:{mime_type};base64,{image_base64}"

    def analyze_image(self, image_path: str, prompt: str = "Describe this image in detail"):
        image_data = self._encode_image(image_path)
        
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_data}
        ])

        response = self.llm.invoke([message])
        return response.content

