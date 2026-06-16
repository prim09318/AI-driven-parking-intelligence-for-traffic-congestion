import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("Key found:", bool(api_key))

genai.configure(api_key=api_key)

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)