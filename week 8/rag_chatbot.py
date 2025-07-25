# rag_chatbot.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Optional LLMs
from transformers import pipeline
import openai

# Optional Gradio UI
try:
    import gradio as gr
except:
    gr = None

# =============================
# 🔹 CONFIGURATION
# =============================
USE_OPENAI = False  # Set to True if you have OpenAI API key
OPENAI_MODEL = "gpt-3.5-turbo"
openai.api_key = os.getenv("OPENAI_API_KEY")

# =============================
# 🔹 STEP 1: Load and Chunk Data
# =============================
def load_chunks_from_csv(path):
    df = pd.read_csv(path)
    chunks = []
    for col in df.columns:
        chunks.append(f"Column '{col}': Sample values: {df[col].dropna().unique()[:5]}")
    chunks.append("Data Summary:\n" + df.describe(include='all').to_string())
    return chunks

# =============================
# 🔹 STEP 2: Embed & Index
# =============================
def create_faiss_index(chunks, model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    vectors = embedder.encode(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return embedder, index, chunks

# =============================
# 🔹 STEP 3: Retrieve
# =============================
def retrieve(query, embedder, index, chunks, top_k=3):
    query_vec = embedder.encode([query])
    _, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]

# =============================
# 🔹 STEP 4: Generate
# =============================
def generate_answer(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    if USE_OPENAI:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        resp = openai.ChatCompletion.create(model=OPENAI_MODEL, messages=messages)
        return resp.choices[0].message.content.strip()
    else:
        local_llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
        return local_llm(prompt, max_new_tokens=200)[0]["generated_text"]

# =============================
# 🔹 STEP 5: Combined RAG QA
# =============================
def rag_answer(question, embedder, index, chunks):
    retrieved = retrieve(question, embedder, index, chunks)
    context = "\n".join(retrieved)
    return generate_answer(context, question)

# =============================
# 🔹 OPTIONAL: CLI Interface
# =============================
def cli_mode():
    chunks = load_chunks_from_csv("Training Dataset.csv")
    embedder, index, chunk_store = create_faiss_index(chunks)
    
    print("RAG Q&A Bot Ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        answer = rag_answer(q, embedder, index, chunk_store)
        print("Bot:", answer)
        print()

# =============================
# 🔹 OPTIONAL: Gradio UI
# =============================
def gradio_mode():
    if gr is None:
        raise ImportError("Install Gradio with `pip install gradio` to use UI")

    chunks = load_chunks_from_csv("Training Dataset.csv")
    embedder, index, chunk_store = create_faiss_index(chunks)

    def respond(question):
        return rag_answer(question, embedder, index, chunk_store)

    gr.Interface(fn=respond, inputs="text", outputs="text", title="Loan Dataset RAG Chatbot").launch()

# =============================
# 🔹 MAIN
# =============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "gradio"], default="cli", help="Run mode: cli or gradio")
    args = parser.parse_args()

    if args.mode == "cli":
        cli_mode()
    elif args.mode == "gradio":
        gradio_mode()
