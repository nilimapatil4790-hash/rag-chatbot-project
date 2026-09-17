import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is missing from .env")

client = InferenceClient(token=token)

# Read PDF
reader = PdfReader("document/sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# Create embeddings
embedded_chunks = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}")

    embedding = client.feature_extraction(
        chunk,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    embedded_chunks.append({
        "text": chunk,
        "embedding": embedding
    })

# Save embeddings
with open("embeddings.json", "w", encoding="utf-8") as file:
    json.dump(embedded_chunks, file)

print("\nAll embeddings created successfully!")
print("Saved to embeddings.json")