import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is missing from .env")

client = InferenceClient(token=token)

text = "Python is a programming language."

embedding = client.feature_extraction(
    text,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding created successfully!")
print("Embedding type:", type(embedding))
print("Embedding:", embedding)