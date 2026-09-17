import os
import json
import numpy as np
import faiss

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is missing from .env")

# Initialize Hugging Face client
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

    context = []

    for i in indices[0]:
        if i != -1:
            context.append(chunks[i])

    return "\n\n".join(context)


def generate_answer(question, context):
    prompt = f"""
Answer the question using only the provided context.

If the answer is not available in the context,
say: "I could not find this information in the document."

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


print("🤖 RAG Chatbot is ready!")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    context = retrieve_context(question)

    answer = generate_answer(question, context)

    print("\nAI:", answer)
    print("-" * 60)