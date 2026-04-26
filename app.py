"""
app.py
The user interface for our YouTube RAG app.
Run with: python -m streamlit run app.py
"""

import streamlit as st
import traceback
from rag_pipeline import (
    get_transcript,
    create_chunks,
    build_vectorstore,
    answer_question,
)

# === PAGE SETUP ===
st.set_page_config(page_title="YouTube RAG", page_icon="🎥")
st.title("🎥 Chat with any YouTube Video")
st.caption("Paste a YouTube URL, then ask questions about the video")


# === SESSION STATE ===
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None


# === SECTION 1: VIDEO INPUT ===
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

# Show which video is currently loaded (helps debugging)
if st.session_state.current_video_id:
    st.info(f"📹 Currently loaded video ID: `{st.session_state.current_video_id}`")

if st.button("Process Video"):
    if not url:
        st.warning("⚠️ Please paste a YouTube URL first")
    else:
        try:
            with st.spinner("Step 1/3: Fetching transcript..."):
                transcript, video_id = get_transcript(url)
                st.write(f"✓ Got transcript with {len(transcript)} segments")
            
            with st.spinner("Step 2/3: Chunking..."):
                documents = create_chunks(transcript, video_id)
                st.write(f"✓ Created {len(documents)} chunks")
            
            with st.spinner("Step 3/3: Building vector store..."):
                st.session_state.vectorstore = build_vectorstore(documents)
                st.session_state.current_video_id = video_id
                st.session_state.messages = []
                st.write("✓ Vector store ready")
            
            st.success(f"✅ Processed video `{video_id}`! Ask me anything below.")
        
        except Exception as e:
            st.error(f"❌ Error processing video:\n\n```\n{type(e).__name__}: {e}\n```")
            st.code(traceback.format_exc(), language="python")


# === SECTION 2: CHAT INTERFACE ===
if st.session_state.vectorstore is not None:
    st.divider()
    
    # Show all past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box for new question
    question = st.chat_input("Ask a question about the video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(st.session_state.vectorstore, question)
                    answer_text = result["answer"] + "\n\n**📍 Sources in video:**\n"

# Deduplicate timestamps and sort them in chronological order
                    unique_timestamps = sorted(set(result["timestamps"]))

                    for ts in unique_timestamps:
                        mins, secs = divmod(ts, 60)
                        link = f"https://www.youtube.com/watch?v={result['video_id']}&t={ts}s"
                        answer_text += f"- [{mins}:{secs:02d}]({link})\n"
                    st.markdown(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                except Exception as e:
                    st.error(f"Error answering: {e}")