"""
rag_pipeline.py
This file is the BRAIN of our RAG system.
It does 4 things: fetch transcript, chunk it, embed it, answer questions.
"""

# === IMPORTS: tools we'll use ===
import os                              # for reading environment variables
import re                              # for pattern matching in URLs
from dotenv import load_dotenv         # to load our secret API key from .env

# YouTube transcript fetcher
from youtube_transcript_api import YouTubeTranscriptApi

# LangChain components - the "RAG framework" that makes our life easy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Load the API key from .env file into the environment
load_dotenv()

# ============================================================
# JOB 1: GET YOUTUBE TRANSCRIPT
# ============================================================

def extract_video_id(url):
    """
    Takes a YouTube URL and returns just the 11-character video ID.
    Example: 'https://youtu.be/ABC123XYZ45' -> 'ABC123XYZ45'
    """
    # This regex pattern matches video IDs after 'v=' or '/'
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")


def get_transcript(url):
    """
    Fetches the full transcript of a YouTube video.
    Returns:
      - transcript: a list of segments (each with text + start time)
      - video_id: the 11-character video ID (used later for timestamps)
    """
    video_id = extract_video_id(url)
    
    # New API (v1.0+): create an instance, then call .fetch()
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id)
    
    # The new API returns a FetchedTranscript object
    # We convert it to the old-style list of dicts for our chunking code
    transcript = [
        {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
        for snippet in fetched
    ]
    
    return transcript, video_id
# ============================================================
# JOB 2: CHUNK THE TRANSCRIPT (preserving timestamps)
# ============================================================

def create_chunks(transcript, video_id):
    """
    Splits the transcript into chunks of ~500 characters each.
    CRITICAL: Each chunk also remembers its start timestamp,
    so we can later show users WHERE in the video the answer is.
    """
    
    # --- Step A: Build one big text + a list of (position, time) pairs ---
    full_text = ""
    timestamps = []  # list of (character_position, seconds_into_video)
    
    for segment in transcript:
        # Record: at this character position in full_text, the video is at this time
        timestamps.append((len(full_text), segment["start"]))
        full_text += segment["text"] + " "
    
    # --- Step B: Use LangChain's splitter to break text into chunks ---
    # It tries to split on sentence boundaries so chunks read naturally
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk ~500 characters
        chunk_overlap=50,    # 50 chars overlap so we don't lose context at edges
    )
    text_chunks = splitter.split_text(full_text)
    
    # --- Step C: For each chunk, find which timestamp it belongs to ---
    documents = []
    position = 0
    for chunk in text_chunks:
        # Find where this chunk starts in the original full_text
        chunk_start_pos = full_text.find(chunk, position)
        
        # Find the nearest timestamp BEFORE this position
        chunk_timestamp = 0
        for pos, ts in timestamps:
            if pos <= chunk_start_pos:
                chunk_timestamp = ts
            else:
                break
        
        # Wrap chunk + metadata in a Document object (LangChain's standard format)
        doc = Document(
            page_content=chunk,
            metadata={
                "timestamp": int(chunk_timestamp),
                "video_id": video_id,
            }
        )
        documents.append(doc)
        position = chunk_start_pos + len(chunk)
    
    return documents
# ============================================================
# JOB 3: CREATE EMBEDDINGS + VECTOR STORE
# ============================================================

def build_vectorstore(documents):
    """
    Builds a fresh vector store for each new video.
    We use a unique collection name based on the video ID so
    different videos don't pollute each other's data.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Use the video_id from the first document's metadata
    # to create a unique collection name (different per video)
    video_id = documents[0].metadata["video_id"]
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=f"video_{video_id}",  # unique per video!
    )
    
    return vectorstore
# ============================================================
# JOB 4: ANSWER QUESTIONS USING THE LLM
# ============================================================

def answer_question(vectorstore, question):
    """
    The actual RAG magic happens here. Steps:
    1. Take the user's question
    2. Find the top 4 most similar chunks from the vector DB
    3. Stuff them into a prompt as context
    4. Send to Llama 3 via Groq
    5. Return the answer + source timestamps for citations
    """
    
    # Step 1: Retrieve top 4 most relevant chunks
    # similarity_search embeds the question and finds closest chunks
    relevant_docs = vectorstore.similarity_search(question, k=4)
    
    # Step 2: Combine the chunk texts into one "context" block
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Step 3: Collect timestamps from the retrieved chunks (for citations)
    timestamps = [doc.metadata["timestamp"] for doc in relevant_docs]
    video_id = relevant_docs[0].metadata["video_id"]
    
    # Step 4: Build the PROMPT - this is where prompt engineering matters
    # We tell the LLM: "Only use the context. Don't make stuff up."
    prompt = f"""You are a helpful assistant answering questions about a YouTube video.
Use ONLY the context below to answer. If the answer isn't in the context,
say "I couldn't find this in the video."

Context from the video:
{context}

Question: {question}

Answer in 2-4 sentences:"""
    
    # Step 5: Call Llama 3 via Groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,  # 0 = factual, 1 = creative. We want factual.
    )
    
    response = llm.invoke(prompt)
    
    # Step 6: Return everything the UI needs
    return {
        "answer": response.content,
        "timestamps": timestamps,
        "video_id": video_id,
    }