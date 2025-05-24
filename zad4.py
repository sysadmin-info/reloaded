#!/usr/bin/env python3
"""
zad4.py - Cenzura danych agentów przez LLM
Cenzuruje imię i nazwisko, wiek, miasto oraz ulicę+numer,
zastępując je słowem "CENZURA" wyłącznie przez LLM.
Obsługa: openai, lmstudio, anything, gemini, claude.
DODANO: Obsługę Claude + liczenie tokenów i kosztów dla wszystkich silników (bezpośrednia integracja)
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
REPORT_URL = os.getenv("REPORT_URL")
ENGINE = os.getenv("LLM_ENGINE", "openai").lower()

# Sprawdzenie czy silnik jest obsługiwany
if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
    sys.exit(1)

print(f"🔄 Engine: {ENGINE}")

# Modele i klucze API
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME_GEMINI = os.getenv("MODEL_NAME_GEMINI", "gemini-2.0-pro-latest")

if not CENTRALA_API_KEY or not REPORT_URL:
    print("❌ Brak ustawienia CENTRALA_API_KEY lub REPORT_URL w .env")
    sys.exit(1)

CENZURA_URL = os.getenv("CENZURA_URL")

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
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ Musisz zainstalować openai: pip install openai", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini"),  # POPRAWKA: użyj modelu OpenAI
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        # Liczenie tokenów (jak w zad1.py i zad2.py)
        tokens = resp.usage
        cost = tokens.prompt_tokens/1_000_000*0.60 + tokens.completion_tokens/1_000_000*2.40
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        return resp.choices[0].message.content.strip()
    
    # --- Claude ---
    elif ENGINE == "claude":
        # Bezpośrednia integracja Claude (jak w zad1.py i zad2.py)
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
        
        resp = claude_client.messages.create(
            model=model_claude,
            messages=[
                {"role": "user", "content": PROMPT_SYSTEM + "\n\n" + prompt_user}
            ],
            system=None,  # system w content
            temperature=0,
            max_tokens=4000
        )
        
        # Liczenie tokenów Claude (jak w zad1.py i zad2.py)
        usage = resp.usage
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015  # Claude Sonnet 4 pricing
        print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")
        
        return resp.content[0].text.strip()
    
    # --- Gemini (Google) ---
    elif ENGINE == "gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            print("❌ Musisz zainstalować google-generativeai: pip install google-generativeai", file=sys.stderr)
            sys.exit(1)
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME_GEMINI)
        response = model.generate_content(
            [PROMPT_SYSTEM + "\n" + prompt_user],
            generation_config={"temperature": 0.0, "max_output_tokens": 4096}
        )
        print(f"[📊 Gemini - brak szczegółów tokenów]")
        print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
        return response.text.strip()
    
    # --- LM Studio / Anything LLM (OpenAI compatible, local) ---
    elif ENGINE in {"lmstudio", "anything"}:
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ Musisz zainstalować openai: pip install openai", file=sys.stderr)
            sys.exit(1)
        api_base = OPENAI_API_URL
        api_key = OPENAI_API_KEY or "local"
        client = OpenAI(api_key=api_key, base_url=api_base)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0
        )
        # Liczenie tokenów (jak w zad1.py i zad2.py)
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
    import os
    url = os.environ.get("CENZURA_URL")
    if not url:
        raise ValueError("CENZURA_URL environment variable is not set")
    raw = download_text(url)
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