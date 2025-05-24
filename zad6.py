#!/usr/bin/env python3
"""
zad6.py - Rozpoznawanie miasta z mapy
DODANO: Obsługę Claude + liczenie tokenów i kosztów dla wszystkich silników (bezpośrednia integracja)
Claude obsługuje vision natywnie
"""
import os
import sys
import base64
from dotenv import load_dotenv

load_dotenv(override=True)

ENGINE = os.getenv("LLM_ENGINE", "openai").lower()
print(f"🔄 Engine: {ENGINE}")

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
        "model": os.getenv("MODEL_NAME_OPENAI", "gpt-4o"),  # POPRAWKA: użyj modelu OpenAI dla vision
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
        result = resp.json()
        # Liczenie tokenów dla OpenAI (jak w zad1.py i zad2.py)
        if "usage" in result:
            usage = result["usage"]
            cost = usage["prompt_tokens"]/1_000_000*0.60 + usage["completion_tokens"]/1_000_000*2.40
            print(f"[📊 Prompt: {usage['prompt_tokens']} | Completion: {usage['completion_tokens']} | Total: {usage['total_tokens']}]")
            print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        return result["choices"][0]["message"]["content"].strip()
    print(f"[Błąd Vision] {resp.status_code}: {resp.text}", file=sys.stderr)
    sys.exit(1)

def call_claude_vision(image_path: str) -> str:
    """Obsługa Claude z vision - Claude obsługuje obrazy natywnie (bezpośrednia integracja)"""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not CLAUDE_API_KEY:
        print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env", file=sys.stderr)
        sys.exit(1)
    
    model_claude = os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    claude_client = Anthropic(api_key=CLAUDE_API_KEY)
    
    # Przygotowanie obrazu dla Claude
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Claude obsługuje obrazy w messages
    resp = claude_client.messages.create(
        model=model_claude,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_SYSTEM + "\n\n" + PROMPT_USER},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }
        ],
        temperature=0.0,
        max_tokens=64
    )
    
    # Liczenie tokenów Claude (jak w zad1.py i zad2.py)
    usage = resp.usage
    cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015  # Claude Sonnet 4 pricing
    print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
    print(f"[💰 Koszt Claude: {cost:.6f} USD]")
    
    return resp.content[0].text.strip()

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
            {"mime_type": "image/jpeg", "data": img_bytes}
        ],
        generation_config={"temperature": 0.0, "max_output_tokens": 64}
    )
    print(f"[📊 Gemini - brak szczegółów tokenów]")
    print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
    return response.text.strip()

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Nie znaleziono pliku obrazu: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if ENGINE == "openai":
            image_b64 = encode_image_to_base64(IMAGE_PATH)
            result = call_openai_vision(image_b64)
        elif ENGINE == "claude":
            result = call_claude_vision(IMAGE_PATH)
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