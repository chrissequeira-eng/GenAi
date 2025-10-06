import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv("myenv/.env")

API_KEY = os.getenv("ELEVENLABS_API_KEY") or "your_api_key_here"
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

def generate_tts(text: str, filename="output.mp3") -> str:
    """Convert text to speech using Eleven Labs API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }

    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename
    else:
        raise Exception(f"TTS Error: {response.status_code} - {response.text}")
