import json
import numpy as np
import faiss
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

import os

token = os.getenv("HF_TOKEN")

client = InferenceClient(token=token)

# Load vector database
index = faiss.read_index("vector_database.index")

# Load text chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Ask a question
question = input("Enter your question: ")

# Create question embedding
embedding = client.feature_extraction(
    question,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

embedding = np.array(embedding, dtype="float32").flatten()
embedding = np.array([embedding], dtype="float32")

# Search similar chunks
distances, indices = index.search(embedding, k=3)

print("\nRelevant information:\n")

for i in indices[0]:
    print(chunks[i])
    print("-" * 50)