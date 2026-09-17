import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is missing from .env")

client = InferenceClient(
    api_key=token
)

response = client.chat_completion(
    model="openai/gpt-oss-120b:fastest",
    messages=[
        {
            "role": "user",
            "content": "Explain what Python is in simple words."
        }
    ],
    max_tokens=200
)

print("LLM connection successful!")
print("\nAI Answer:\n")
print(response.choices[0].message.content)