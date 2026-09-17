import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is missing from .env")

api = HfApi(token=token)

user_info = api.whoami()

print("Hugging Face connection successful!")
print("Username:", user_info.get("name"))