import os
import json
import numpy as np
import faiss
import streamlit as st

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variablesload_dotenv()

try:
    token = st.secrets["HF_TOKEN"]
except Exception:
    token = os.getenv("HF_TOKEN")

if not token:
    st.error("HF_TOKEN is missing.")
    st.stop()

if not token:
    st.error("HF_TOKEN is missing from .env")
    st.stop()

# Initialize client
client = InferenceClient(api_key=token)

# Load vector database
index = faiss.read_index("vector_database.index")

# Load text chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


def create_embedding(text):
    embedding = client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    return np.array(embedding, dtype="float32").flatten()


def retrieve_context(question, k=3):
    question_embedding = create_embedding(question)

    question_embedding = np.array(
        [question_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        question_embedding,
        k
    )

    relevant_chunks = []

    for i in indices[0]:
        if i != -1:
            relevant_chunks.append(chunks[i])

    return "\n\n".join(relevant_chunks)


def generate_answer(question, context):
    prompt = f"""
Answer the question using only the provided context.

If the answer is not available in the context,
say that the information was not found in the document.

Context:
{context}

Question:
{question}
"""

    response = client.chat_completion(
        model="openai/gpt-oss-120b:fastest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300
    )

    return response.choices[0].message.content


# Streamlit interface
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 PDF RAG Chatbot")
st.write("Ask questions about your PDF document.")

question = st.text_input("Enter your question:")

if st.button("Ask AI"):
    if question.strip():
        with st.spinner("Searching and generating answer..."):
            context = retrieve_context(question)
            answer = generate_answer(question, context)

        st.subheader("AI Answer")
        st.write(answer)

    else:
        st.warning("Please enter a question.")