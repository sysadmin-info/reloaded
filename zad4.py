#!/usr/bin/env python3
"""
S01E04 - Cenzura danych agentów przez LLM
Cenzuruje imię i nazwisko, wiek, miasto oraz ulicę+numer,
zastępując je słowem "CENZURA" wyłącznie przez LLM.
Obsługa: openai, lmstudio, anything, gemini, claude.
DODANO: Obsługę Claude + liczenie tokenów i kosztów dla wszystkich silników (bezpośrednia integracja)
POPRAWKA: Lepsze wykrywanie silnika z agent.py
"""

import argparse
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# POPRAWKA: Dodano argumenty CLI jak w innych zadaniach
parser = argparse.ArgumentParser(description="Cenzura danych (multi-engine + Claude)")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
args = parser.parse_args()

# POPRAWKA: Lepsze wykrywanie silnika (jak w poprawionych zad1.py i zad2.py)
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
            ENGINE = "lmstudio"  # domyślnie

print(f"🔄 ENGINE wykryty: {ENGINE}")

# Sprawdzenie czy silnik jest obsługiwany
if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
    sys.exit(1)

print(f"✅ Engine: {ENGINE}")

CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
REPORT_URL = os.getenv("REPORT_URL")
CENZURA_URL = os.getenv("CENZURA_URL")

if not CENTRALA_API_KEY or not REPORT_URL or not CENZURA_URL:
    print("❌ Brak ustawienia CENTRALA_API_KEY, REPORT_URL lub CENZURA_URL w .env")
    sys.exit(1)

def download_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text.strip()
    except Exception as e:
        print(f"❌ Błąd podczas pobierania danych: {e}")
        sys.exit(1)

# --- ULTRA-TWARDY PROMPT ---
PROMPT_SYSTEM = (
    "Jesteś automatem do cenzury danych osobowych w języku polskim. "
    "NIE WOLNO Ci zmieniać żadnych innych słów, znaków interpunkcyjnych, układu tekstu ani zamieniać kolejności zdań. "
    "Zamień TYLKO i WYŁĄCZNIE:\n"
    "- każde imię i nazwisko na 'CENZURA',\n"
    "- każdą nazwę miasta na 'CENZURA',\n"
    "- każdą nazwę ulicy wraz z numerem domu/mieszkania na 'CENZURA',\n"
    "- każdą informację o wieku (np. '45 lat', 'wiek: 32', 'lat 27', 'ma 29 lat') na 'CENZURA'.\n"
    "Nie wolno parafrazować, nie wolno podsumowywać, nie wolno streszczać ani zamieniać kolejności czegokolwiek. "
    "Wynikowy tekst musi mieć identyczny układ, interpunkcję i liczbę linii jak oryginał. "
    "Każda inna zmiana niż cenzura wyżej powoduje błąd i NIEZALICZENIE zadania. "
    "Nie pisz żadnych komentarzy, nie wyjaśniaj odpowiedzi. "
    "ODPOWIEDZ WYŁĄCZNIE TEKSTEM Z OCENZURĄ. "
    "PRZYKŁAD:\n"
    "Oryginał:\n"
    "Dane podejrzanego: Jan Kowalski, lat 45, mieszka w Krakowie, ul. Polna 8.\n"
    "Wyjście:\n"
    "Dane podejrzanego: CENZURA, lat CENZURA, mieszka w CENZURA, ul. CENZURA."
)

def censor_llm(text: str) -> str:
    prompt_user = (
        "Tekst do cenzury (nie zmieniaj nic poza danymi osobowymi, przykład wyżej!):\n"
        + text
    )
    
    # --- OpenAI ---
    if ENGINE == "openai":
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
        
        if not OPENAI_API_KEY:
            print("❌ Brak OPENAI_API_KEY", file=sys.stderr)
            sys.exit(1)
            
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ Musisz zainstalować openai: pip install openai", file=sys.stderr)
            sys.exit(1)
            
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        # Liczenie tokenów
        tokens = resp.usage
        cost = tokens.prompt_tokens/1_000_000*0.60 + tokens.completion_tokens/1_000_000*2.40
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        return resp.choices[0].message.content.strip()
    
    # --- Claude ---
    elif ENGINE == "claude":
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
        
        if not CLAUDE_API_KEY:
            print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env", file=sys.stderr)
            sys.exit(1)
            
        try:
            from anthropic import Anthropic
        except ImportError:
            print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        
        claude_client = Anthropic(api_key=CLAUDE_API_KEY)
        resp = claude_client.messages.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": PROMPT_SYSTEM + "\n\n" + prompt_user}
            ],
            temperature=0,
            max_tokens=4000
        )
        
        # Liczenie tokenów Claude
        usage = resp.usage
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
        print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")
        
        return resp.content[0].text.strip()
    
    # --- Gemini (Google) ---
    elif ENGINE == "gemini":
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
        
        if not GEMINI_API_KEY:
            print("❌ Brak GEMINI_API_KEY w .env", file=sys.stderr)
            sys.exit(1)
            
        try:
            import google.generativeai as genai
        except ImportError:
            print("❌ Musisz zainstalować google-generativeai: pip install google-generativeai", file=sys.stderr)
            sys.exit(1)
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            [PROMPT_SYSTEM + "\n" + prompt_user],
            generation_config={"temperature": 0.0, "max_output_tokens": 4096}
        )
        print(f"[📊 Gemini - brak szczegółów tokenów]")
        print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
        return response.text.strip()
    
    # --- LM Studio ---
    elif ENGINE == "lmstudio":
        LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "local")
        LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
        MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
        
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ Musisz zainstalować openai: pip install openai", file=sys.stderr)
            sys.exit(1)
            
        client = OpenAI(api_key=LMSTUDIO_API_KEY, base_url=LMSTUDIO_API_URL)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        # Liczenie tokenów
        tokens = resp.usage
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        print(f"[💰 Model lokalny - brak kosztów]")
        return resp.choices[0].message.content.strip()
    
    # --- Anything LLM ---
    elif ENGINE == "anything":
        ANYTHING_API_KEY = os.getenv("ANYTHING_API_KEY", "local")
        ANYTHING_API_URL = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")
        
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ Musisz zainstalować openai: pip install openai", file=sys.stderr)
            sys.exit(1)
            
        client = OpenAI(api_key=ANYTHING_API_KEY, base_url=ANYTHING_API_URL)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        # Liczenie tokenów
        tokens = resp.usage
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        print(f"[💰 Model lokalny - brak kosztów]")
        return resp.choices[0].message.content.strip()
    else:
        print(f"❌ Nieznany silnik: {ENGINE}", file=sys.stderr)
        sys.exit(1)

def extract_flag(text: str) -> str:
    import re
    m = re.search(r"\{\{FLG:[^}]+\}\}|FLG\{[^}]+\}", text)
    return m.group(0) if m else ""

def main():
    raw = download_text(CENZURA_URL)
    print(f"🔄 Pobrano tekst ({len(raw)} znaków)")
    print(f"🔄 Cenzuruję używając {ENGINE}...")
    
    censored = censor_llm(raw)
    print("=== OCENZUROWANY OUTPUT ===")
    print(censored)
    print("===========================")
    
    payload = {
        "task": "CENZURA",
        "apikey": CENTRALA_API_KEY,
        "answer": censored
    }
    try:
        r = requests.post(REPORT_URL, json=payload, timeout=10)
        if r.ok:
            resp_text = r.text.strip()
            flag = extract_flag(resp_text) or extract_flag(censored)
            if flag:
                print(flag)
            else:
                print("Brak flagi w odpowiedzi serwera. Odpowiedź:", resp_text)
        else:
            print(f"❌ Błąd HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print(f"❌ Błąd podczas wysyłania danych: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()