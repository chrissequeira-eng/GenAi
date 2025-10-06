import os
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from STT import transcribe_audio
from TTS import generate_tts

# Load env vars
load_dotenv("myenv/.env")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Prompt for evaluation
prompt_template = """
You are an AI interviewer evaluating a candidate’s spoken answer.
Evaluate the candidate’s response on:
- Clarity
- Structure
- Correctness
- Confidence
- Tone

Then give constructive feedback and a short summary as if you were a real interviewer.
Answer in a friendly and professional tone.
Candidate's answer:
{answer}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

def evaluate_answer(transcript: str) -> str:
    """Use Gemini to evaluate the transcribed answer."""
    if not transcript.strip():
        return "No answer provided."
    messages = prompt.format_messages(answer=transcript)
    response = llm(messages)
    return response.content.strip()

# ---------- Streamlit UI ----------
def main():
    st.title("🎯 AI Interview Coach")
    st.info("Speak your answer to an interview question. The AI will evaluate you and then *speak* its feedback!")

    # Step 1 — Record Audio
    st.header("Step 1: Record your answer")
    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop",
        key="recorder"
    )

    transcript = ""
    feedback = ""

    if audio and audio.get("bytes"):
        st.info("⏳ Transcribing your answer...")
        transcript = transcribe_audio(audio["bytes"])
        st.success("✅ Transcription complete")
        st.text_area("Your Transcript", transcript, height=200)

        # Step 2 — Evaluate with LLM
        st.info("🤖 Evaluating your answer...")
        feedback = evaluate_answer(transcript)
        st.success("✅ Evaluation complete")
        st.text_area("AI Feedback", feedback, height=200)

        # Step 3 — Speak feedback (TTS)
        st.info("🔊 Generating spoken feedback...")
        try:
            audio_file = generate_tts(feedback)
            st.audio(audio_file, format="audio/mp3")
        except Exception as e:
            st.error(f"TTS Error: {e}")

        # Optional session history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"answer": transcript, "feedback": feedback})

    # Display session history
    if "history" in st.session_state and st.session_state.history:
        st.subheader("🧾 Session History")
        for i, entry in enumerate(st.session_state.history):
            st.write(f"**Answer {i+1}:** {entry['answer']}")
            st.write(f"**Feedback:** {entry['feedback']}")
            st.markdown("---")

if __name__ == "__main__":
    main()
