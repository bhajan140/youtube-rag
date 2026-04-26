# 🎥 YouTube RAG — Chat with any YouTube Video

An AI-powered application that lets you **ask questions about any YouTube video** and get answers grounded in the actual content — with **clickable timestamps** that jump to the exact moment in the video.

Built with **LangChain**, **ChromaDB**, **Groq (Llama 3)**, and **Streamlit**.

---

## ✨ Features

- 🔗 Paste any YouTube URL → automatically fetches the transcript
- 💬 Ask questions in natural language about the video content
- 📍 Answers include **clickable timestamps** that jump to the exact moment in the video
- 🛡️ Hallucination-resistant: answers are grounded only in the video content
- ⚡ Sub-second response times via Groq's LPU inference

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.1 (via Groq API) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| UI | Streamlit |
| Transcript | youtube-transcript-api |

---

## 🧠 How It Works
YouTube URL  →  Transcript  →  Chunks  →  Embeddings  →  ChromaDB
↓
User Question  →  Embed  →  Retrieve top-4 similar chunks  ←──┘
↓
Augment prompt with context
↓
Llama 3 generates grounded answer + timestamp citations

The pipeline does two things:
1. **Indexing** (one-time per video): fetches transcript, chunks it while preserving timestamps as metadata, embeds chunks, stores in ChromaDB.
2. **Querying**: retrieves top-4 most similar chunks for the user's question, augments the LLM prompt with this context, generates an answer grounded in the video.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- A free [Groq API key](https://console.groq.com)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/youtube-rag.git
cd youtube-rag

# Create environment
conda create -n ragenv python=3.11 -y
conda activate ragenv

# Install dependencies
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Then edit .env and paste your Groq API key
```

### Run

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure
youtube-rag/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG logic (transcript, chunking, embedding, retrieval, generation)
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .gitignore          # Hidden files
└── README.md           # This file

---

## 🐛 Engineering Challenges Solved

- **ChromaDB collection isolation**: switched to `collection_name` per video to prevent cross-video contamination
- **Timestamp metadata preservation**: built custom logic to map character positions back to video timestamps after chunking
- **Stale session state in Streamlit**: implemented proper state reset when new videos are loaded
- **Secure API key handling**: used `python-dotenv` and `.gitignore` to keep credentials out of source control

---

## 📜 License

MIT