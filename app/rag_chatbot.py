import streamlit as st
import os
import io
import base64
from groq import Groq
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
from rag_engine import retrieve_relevant_docs


# ── Groq client ───────────────────────────────────────────────────────────────
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return Groq(api_key=api_key), api_key


# ── Speech to Text via Groq Whisper ──────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio bytes to Groq Whisper for transcription."""
    client, _ = get_groq_client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
        response_format="text"
    )
    return transcription.strip()


# ── Text to Speech via gTTS ───────────────────────────────────────────────────
def text_to_speech(text: str) -> str:
    """Convert text to speech and return base64 audio string."""
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    audio_base64 = base64.b64encode(mp3_buffer.read()).decode('utf-8')
    return audio_base64


def autoplay_audio(audio_base64: str):
    """Play audio using Streamlit native audio player."""
    import base64 as b64lib
    audio_bytes = b64lib.b64decode(audio_base64)
    buf = io.BytesIO(audio_bytes)
    st.audio(buf, format="audio/mp3", autoplay=True)


# ── Build RAG prompt ──────────────────────────────────────────────────────────
def build_rag_prompt(question: str, retrieved_docs: list) -> str:
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_blocks.append(
            f"[Document {i}: {doc['title']}]\n{doc['content']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"""You are an AI assistant for IntelliRisk AI, an enterprise financial risk platform.
You help users understand loan approval decisions and insurance fraud detection results.

You have been given the following relevant policy documents to answer the question:

{context}

---

Using ONLY the information from the documents above, answer this question clearly and helpfully.
If the documents don't contain enough information, say so honestly.
Always mention which policy or rule you are referencing.
Keep answers concise — 2 to 4 sentences. This will be read aloud so avoid bullet points and markdown.

Question: {question}"""


# ── Ask Groq with RAG ─────────────────────────────────────────────────────────
def ask_groq_with_rag(question: str) -> tuple:
    retrieved_docs = retrieve_relevant_docs(question, n_results=3)
    prompt = build_rag_prompt(question, retrieved_docs)
    client, _ = get_groq_client()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=512,
        temperature=0.3,
    )
    answer = chat_completion.choices[0].message.content
    return answer, retrieved_docs


# ── Main Chatbot UI ───────────────────────────────────────────────────────────
def render_rag_chatbot():

    st.markdown("""
    <style>
    .chat-source {
        background: rgba(29, 158, 117, 0.08);
        border-left: 3px solid #1D9E75;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 12px;
        color: #8899aa;
    }
    .chat-source strong {
        color: #1D9E75;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .powered-badge {
        display: inline-block;
        background: rgba(83,74,183,0.15);
        border: 1px solid rgba(83,74,183,0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        color: #AFA9EC;
        margin-left: 8px;
        vertical-align: middle;
    }
    .voice-tip {
        background: rgba(29,158,117,0.08);
        border: 1px solid rgba(29,158,117,0.2);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 12px;
        color: #8899aa;
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "## IntelliRisk AI Chatbot "
        "<span class='powered-badge'>⚡ Llama 3.3 · RAG · ChromaDB</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#8899aa;margin-top:-10px;margin-bottom:16px'>"
        "Ask anything about loan approvals, fraud detection rules, or risk policies. "
        "Type or use your voice.</p>",
        unsafe_allow_html=True
    )

    # ── API key check ─────────────────────────────────────────────────────────
    _, api_key = get_groq_client()
    if not api_key:
        st.error("GROQ_API_KEY not found.")
        st.markdown("Run this in your terminal then restart Streamlit:")
        st.code('export GROQ_API_KEY="gsk_your-key-here"', language="bash")
        return

    # ── Session state ─────────────────────────────────────────────────────────
    if "rag_messages"     not in st.session_state:
        st.session_state.rag_messages = []
    if "rag_sources"      not in st.session_state:
        st.session_state.rag_sources = {}
    if "voice_enabled"    not in st.session_state:
        st.session_state.voice_enabled = True
    if "last_audio_key"   not in st.session_state:
        st.session_state.last_audio_key = None

    # ── Voice toggle + mic row ────────────────────────────────────────────────
    col_toggle, col_mic = st.columns([1, 3])

    with col_toggle:
        st.session_state.voice_enabled = st.toggle(
            "🔊 Read answers aloud",
            value=st.session_state.voice_enabled
        )

    with col_mic:
        st.markdown(
            "<div class='voice-tip'>"
            "🎤 Click the mic button below to speak your question — "
            "it will be transcribed automatically."
            "</div>",
            unsafe_allow_html=True
        )

    # ── Mic recorder ─────────────────────────────────────────────────────────
    audio = mic_recorder(
        start_prompt="🎤 Click to speak",
        stop_prompt="⏹ Stop recording",
        just_once=True,
        use_container_width=True,
        key="mic_recorder"
    )

    # ── Process voice input ───────────────────────────────────────────────────
    if audio and audio.get("bytes"):
        audio_key = audio.get("id", str(len(audio["bytes"])))
        if audio_key != st.session_state.last_audio_key:
            st.session_state.last_audio_key = audio_key
            with st.spinner("🎤 Transcribing your voice..."):
                try:
                    transcribed = transcribe_audio(audio["bytes"])
                    if transcribed:
                        st.success(f'🎤 You said: *"{transcribed}"*')
                        # Process as a question
                        st.session_state.rag_messages.append(
                            {"role": "user", "content": transcribed}
                        )
                        with st.spinner("Searching knowledge base..."):
                            answer, sources = ask_groq_with_rag(transcribed)
                        msg_idx = len(st.session_state.rag_messages)
                        st.session_state.rag_messages.append(
                            {"role": "assistant", "content": answer}
                        )
                        st.session_state.rag_sources[msg_idx] = sources
                        # Auto play audio response
                        if st.session_state.voice_enabled:
                            audio_b64 = text_to_speech(answer)
                            autoplay_audio(audio_b64)
                        st.rerun()
                except Exception as e:
                    st.error(f"Transcription error: {e}")

    st.divider()

    # ── Suggested questions ───────────────────────────────────────────────────
    if not st.session_state.rag_messages:
        st.markdown(
            "<p style='font-size:13px;color:#8899aa;margin-bottom:8px'>"
            "Try asking:</p>",
            unsafe_allow_html=True
        )
        suggestions = [
            "Why would a loan be rejected?",
            "What credit score is needed for approval?",
            "Why is policy holder fault more suspicious?",
            "How does the anomaly detection score work?",
            "What is a high-risk loan applicant?",
            "What happens when fraud is confirmed?",
        ]
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 3].button(suggestion, key=f"sug_{i}"):
                st.session_state.rag_messages.append(
                    {"role": "user", "content": suggestion}
                )
                with st.spinner("Searching knowledge base..."):
                    answer, sources = ask_groq_with_rag(suggestion)
                msg_idx = len(st.session_state.rag_messages)
                st.session_state.rag_messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.rag_sources[msg_idx] = sources
                if st.session_state.voice_enabled:
                    audio_b64 = text_to_speech(answer)
                    autoplay_audio(audio_b64)
                st.rerun()
        st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────────
    for idx, message in enumerate(st.session_state.rag_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and idx in st.session_state.rag_sources:
                sources = st.session_state.rag_sources[idx]
                with st.expander(f"📚 Sources used ({len(sources)} documents)", expanded=False):
                    for src in sources:
                        st.markdown(
                            f"<div class='chat-source'>"
                            f"<strong>Source</strong><br>{src['title']}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # ── Text chat input ───────────────────────────────────────────────────────
    if user_input := st.chat_input("Or type your question here..."):
        st.session_state.rag_messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                answer, sources = ask_groq_with_rag(user_input)
            st.markdown(answer)

            msg_idx = len(st.session_state.rag_messages)
            st.session_state.rag_sources[msg_idx] = sources

            with st.expander(f"📚 Sources used ({len(sources)} documents)", expanded=False):
                for src in sources:
                    st.markdown(
                        f"<div class='chat-source'>"
                        f"<strong>Source</strong><br>{src['title']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            if st.session_state.voice_enabled:
                with st.spinner("🔊 Generating audio..."):
                    audio_b64 = text_to_speech(answer)
                autoplay_audio(audio_b64)

        st.session_state.rag_messages.append(
            {"role": "assistant", "content": answer}
        )

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.session_state.rag_messages:
        if st.button("🗑️ Clear chat", type="secondary"):
            st.session_state.rag_messages = []
            st.session_state.rag_sources = {}
            st.rerun()