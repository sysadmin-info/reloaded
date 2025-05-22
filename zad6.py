#!/usr/bin/env python3
import os
import sys
import base64
from dotenv import load_dotenv

ENGINE = os.getenv("LLM_ENGINE", "openai").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IMAGE_PATH = os.getenv("MAP_IMAGE_PATH")

PROMPT_SYSTEM = (
    "Jesteś ekspertem od polskich map i kartografii. "
    "Analizujesz fragmenty mapy przedstawiające miasto ze spichlerzami i twierdzą,"
    "położone nad Wisłą w północnej Polsce. Nie jest to Toruń ani Chełmno."
    "Jeden fragment mapy może być przypadkowy. Odpowiedz wyłącznie nazwą miasta."
)
PROMPT_USER = "Jakie miasto przedstawiają te fragmenty mapy? Jeden z nich może być przypadkowy. Podaj tylko nazwę miasta. Odpowiedz tylko w formacie {{FLG:...}}."

def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_flag(text: str) -> str:
    import re
    match = re.search(r"\{\{FLG:[^}]+\}\}|FLG\{[^}]+\}", text)
    return match.group(0) if match else ""

def call_openai_vision(image_b64: str) -> str:
    import requests
    url = f"{OPENAI_API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_USER},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.ok:
        return resp.json()["choices"][0]["message"]["content"].strip()
    print(f"[Błąd Vision] {resp.status_code}: {resp.text}", file=sys.stderr)
    sys.exit(1)

def call_gemini_vision(image_path: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        print("Musisz zainstalować google-generativeai: pip install google-generativeai", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    with open(image_path, "rb") as imgf:
        img_bytes = imgf.read()
    response = model.generate_content(
        [
            PROMPT_SYSTEM + "\n" + PROMPT_USER,
            {"mime_type": "image/jpeg", "data": img_bytes}  # <-- poprawka
        ],
        generation_config={"temperature": 0.0, "max_output_tokens": 64}
    )
    return response.text.strip()

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Nie znaleziono pliku obrazu: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)
    image_b64 = encode_image_to_base64(IMAGE_PATH)
    try:
        if ENGINE == "openai":
            result = call_openai_vision(image_b64)
        elif ENGINE in {"lmstudio", "anything"}:
            print("Model lokalny nie obsługuje multimodalnych wejść.", file=sys.stderr)
            sys.exit(1)
        elif ENGINE == "gemini":
            result = call_gemini_vision(IMAGE_PATH)
        else:
            print(f"Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
            sys.exit(1)
        flag = extract_flag(result)
        if flag:
            print(flag)
            sys.exit(0)
        else:
            print("Nie znaleziono flagi.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Błąd: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
