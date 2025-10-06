import os
from dotenv import load_dotenv
from deepgram import DeepgramClient

# Load environment variables
load_dotenv("myenv/.env")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Initialize client
dg_client = DeepgramClient(DEEPGRAM_API_KEY)

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe recorded audio using Deepgram."""
    try:
        response = dg_client.listen.rest.v("1").transcribe_file(
            {"buffer": audio_bytes},
            {"model": "nova-2-general", "language": "en-US", "smart_format": True}
        )
        transcript = response.results.channels[0].alternatives[0].transcript
        transcript = transcript.strip() or "No transcript generated."

        # Save for reference
        with open("transcript.txt", "w") as f:
            f.write(transcript)

        return transcript
    except Exception as e:
        return f"Transcription error: {e}"
