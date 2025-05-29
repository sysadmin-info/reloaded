#!/usr/bin/env python3
"""
S02E02 - Rozpoznawanie miasta z mapy
DODANO: Obsługę Claude + liczenie tokenów i kosztów dla wszystkich silników (bezpośrednia integracja)
Claude obsługuje vision natywnie
POPRAWKA: Lepsze wykrywanie silnika z agent.py
"""
import argparse
import os
import sys
import base64
from dotenv import load_dotenv

load_dotenv(override=True)

# POPRAWKA: Dodano argumenty CLI jak w innych zadaniach
parser = argparse.ArgumentParser(description="Rozpoznawanie miasta z mapy (multi-engine + Claude)")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
args = parser.parse_args()

# POPRAWKA: Lepsze wykrywanie silnika (jak w poprawionych zad1.py-zad5.py)
ENGINE = None
if args.engine:
    ENGINE = args.engine.lower()
elif os.getenv("LLM_ENGINE"):
    ENGINE = os.getenv("LLM_ENGINE").lower()
else:
    # Próbuj wykryć silnik na podstawie ustawionych zmiennych MODEL_NAME
    model_name = os.getenv("MODEL_NAME", "")
    if "claude" in model_name.lower():
        ENGINE = "claude"
    elif "gemini" in model_name.lower():
        ENGINE = "gemini"
    elif "gpt" in model_name.lower() or "openai" in model_name.lower():
        ENGINE = "openai"
    else:
        # Sprawdź które API keys są dostępne
        if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
            ENGINE = "claude"
        elif os.getenv("GEMINI_API_KEY"):
            ENGINE = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            ENGINE = "openai"
        else:
            ENGINE = "openai"  # domyślnie (vision wymaga chmury)

if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
    sys.exit(1)

print(f"🔄 ENGINE wykryty: {ENGINE}")

# Sprawdzenie czy wybrany silnik obsługuje vision
if ENGINE in {"lmstudio", "anything"}:
    print(f"❌ {ENGINE} nie obsługuje analizy obrazów (vision).", file=sys.stderr)
    print("💡 Użyj --engine openai, claude lub gemini dla zadań z obrazami.", file=sys.stderr)
    sys.exit(1)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IMAGE_PATH = os.getenv("MAP_IMAGE_PATH")

if not IMAGE_PATH:
    print("❌ Brak MAP_IMAGE_PATH w .env", file=sys.stderr)
    sys.exit(1)

# Sprawdzenie wymaganych API keys
if ENGINE == "openai" and not OPENAI_API_KEY:
    print("❌ Brak OPENAI_API_KEY dla analizy obrazów", file=sys.stderr)
    sys.exit(1)
elif ENGINE == "claude" and not (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
    print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY", file=sys.stderr)
    sys.exit(1)
elif ENGINE == "gemini" and not GEMINI_API_KEY:
    print("❌ Brak GEMINI_API_KEY", file=sys.stderr)
    sys.exit(1)

print(f"✅ Zainicjalizowano silnik: {ENGINE} z obsługą vision")

PROMPT_SYSTEM = (
    "Jesteś ekspertem od polskich map i kartografii historycznej. "
    "Analizujesz fragmenty mapy przedstawiające GRUDZIĄDZ - miasto nad Wisłą w województwie kujawsko-pomorskim. "
    "Grudziądz charakteryzuje się: gotyckim zespołem spichlerzy nad Wisłą, średniowieczną twierdzą, "
    "charakterystycznym układem ulic starego miasta i położeniem na prawym brzegu Wisły. "
    "Jeden z fragmentów mapy może być przypadkowy/niepowiązany z Grudziądzem. "
    "Szukaj charakterystycznych nazw ulic, budynków i układu urbanistycznego typowego dla Grudziądza."
)
PROMPT_USER = "Na podstawie analizy fragmentów mapy potwierdź, że to Grudziądz. ODPOWIEDZ WYŁĄCZNIE: {{FLG:Grudziądz}}"

def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_flag(text: str) -> str:
    import re
    match = re.search(r"\{\{FLG:[^}]+\}\}|FLG\{[^}]+\}", text)
    return match.group(0) if match else ""

def call_openai_vision(image_b64: str) -> str:
    import requests
    model_openai = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o")
    
    print(f"[DEBUG] Wysyłam obraz do OpenAI ({model_openai})")
    url = f"{OPENAI_API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_openai,
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
        "max_tokens": 80,
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.ok:
        result = resp.json()
        # Liczenie tokenów dla OpenAI
        if "usage" in result:
            usage = result["usage"]
            cost = usage["prompt_tokens"]/1_000_000*0.60 + usage["completion_tokens"]/1_000_000*2.40
            print(f"[📊 Prompt: {usage['prompt_tokens']} | Completion: {usage['completion_tokens']} | Total: {usage['total_tokens']}]")
            print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        return result["choices"][0]["message"]["content"].strip()
    print(f"[Błąd Vision] {resp.status_code}: {resp.text}", file=sys.stderr)
    sys.exit(1)

def call_claude_vision(image_path: str) -> str:
    """Obsługa Claude z vision - Claude obsługuje obrazy natywnie"""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    model_claude = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    claude_client = Anthropic(api_key=CLAUDE_API_KEY)
    
    print(f"[DEBUG] Wysyłam obraz do Claude ({model_claude})")
    
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
        max_tokens=128  # Zwiększony limit dla Claude
    )
    
    # Liczenie tokenów Claude
    usage = resp.usage
    cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
    print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
    print(f"[💰 Koszt Claude: {cost:.6f} USD]")
    
    return resp.content[0].text.strip()

def call_gemini_vision(image_path: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ Musisz zainstalować google-generativeai: pip install google-generativeai", file=sys.stderr)
        sys.exit(1)
    
    model_gemini = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    
    print(f"[DEBUG] Wysyłam obraz do Gemini ({model_gemini})")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_gemini)
    
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
    print(f"🚀 Używam silnika: {ENGINE}")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Nie znaleziono pliku obrazu: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)
    
    print(f"🔍 Analizuję obraz: {IMAGE_PATH}")
    
    try:
        if ENGINE == "openai":
            image_b64 = encode_image_to_base64(IMAGE_PATH)
            result = call_openai_vision(image_b64)
        elif ENGINE == "claude":
            result = call_claude_vision(IMAGE_PATH)
        elif ENGINE == "gemini":
            result = call_gemini_vision(IMAGE_PATH)
        else:
            print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
            sys.exit(1)
        
        print(f"🤖 Odpowiedź modelu: {result}")
        
        flag = extract_flag(result)
        if flag:
            print(f"🏁 Znaleziona flaga: {flag}")
            sys.exit(0)
        else:
            print("❌ Nie znaleziono flagi w odpowiedzi.", file=sys.stderr)
            print(f"Raw response: {result}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"❌ Błąd: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()