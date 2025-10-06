AI Interview Coach

A Streamlit app that simulates a real interview experience. Users can record answers, which are transcribed via Deepgram, evaluated by Gemini LLM, and played back as audio feedback using ElevenLabs TTS.

Features
* Record and transcribe answers (STT)
* AI evaluates answers for clarity, structure, tone, and confidence
* Provides spoken feedback (TTS)
* Session history of answers and feedback

Dependencies
streamlit, python-dotenv, requests, deepgram-sdk, langchain, langchain-google-genai, streamlit-mic-recorder, pydub, openai
