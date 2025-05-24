#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S02E02 (multi-engine + Claude)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• zgodny z openai-python ≥ 1.0
• obsługuje backendy: openai / lmstudio / anything (LocalAI) / gemini (Google) / claude (Anthropic)
• nadal celowo „kłamie" (Kraków, 69, 1999, blue) - logika oryginału zachowana.
DODANO: Obsługę Claude z kompatybilnym interfejsem
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from typing import Dict, Tuple

import requests
import urllib3
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import Claude integration (opcjonalny)
try:
    from claude_integration import setup_claude_for_task, add_token_counting_to_openai_call
except ImportError:
    # Kontynuujemy bez Claude - brak komunikatu o błędzie
    pass

# ── 0. CLI / env - wybór silnika ─────────────────────────────────────────────
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Android bot (multi-engine + Claude)")
parser.add_argument("--engine",
                    choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
args = parser.parse_args()

ENGINE = (args.engine or os.getenv("LLM_ENGINE", "openai")).lower()

# ── 1. konfiguracja robota ───────────────────────────────────────────────────
ROBOT_BASE_URL = os.getenv("ROBOT_LOGIN_URL", "").rstrip("/")
VERIFY_URL     = urllib.parse.urljoin(ROBOT_BASE_URL, "/verify")
USERNAME       = os.getenv("ROBOT_USERNAME", "")
PASSWORD       = os.getenv("ROBOT_PASSWORD", "")
HDRS = {"Accept": "application/json"}

# ── 2. fałszywe odpowiedzi + wzorce ──────────────────────────────────────────
FALSE_ANSWERS: Dict[str, Dict[str, str]] = {
    "capital": {"pl": "Kraków",  "en": "Krakow",  "fr": "Krakow"},
    "42":      {"pl": "69",      "en": "69",      "fr": "69"},
    "year":    {"pl": "1999",    "en": "1999",    "fr": "1999"},
    "sky":     {"pl": "blue",    "en": "blue",    "fr": "blue"},
}

PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?:capital|stolic\\w?).*pol",                       re.I), "capital"),
    (re.compile(r"(?:meaning|answer).*life|autostopem|hitchhiker",   re.I), "42"),
    (re.compile(r"(?:what|current).*year|jaki.*rok|ann[ée]e",        re.I), "year"),
    (re.compile(r"(?:colou?r|couleur).*sky|niebo|ciel",              re.I), "sky"),
)

# ── 3. detekcja języka ───────────────────────────────────────────────────────
FR_HINT = re.compile(r"[àâçéèêëîïôûùüÿœ]|\\bcouleur|\\bciel", re.I)
PL_HINT = re.compile(r"[ąćęłńóśżź]|jakiego|który|jaki",      re.I)

def detect_lang(text: str) -> str:
    if PL_HINT.search(text):
        return "pl"
    if FR_HINT.search(text):
        return "fr"
    return "en"

# ── 4. odpowiedzi ────────────────────────────────────────────────────────────
def answer_locally(question: str) -> str | None:
    lang = detect_lang(question)
    for rx, key in PATTERNS:
        if rx.search(question):
            return FALSE_ANSWERS[key][lang]
    return None

SYSTEM_PROMPTS = {
    "pl": "Odpowiadasz bardzo krótko i wyłącznie po polsku (max 2 słowa).",
    "fr": "Réponds très brièvement en français (max 2 mots).",
    "en": "Answer very concisely in English (max 2 words).",
}

# ── 5. Inicjalizacja klienta LLM ──────────────────────────────────────────────
if ENGINE == "openai":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)

elif ENGINE == "lmstudio":
    LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "local")
    LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
    MODEL_NAME = os.getenv("MODEL_NAME_LM", os.getenv("MODEL_NAME", "llama-3.3-70b-instruct"))
    print(f"[DEBUG] LMStudio URL: {LMSTUDIO_API_URL}")
    print(f"[DEBUG] LMStudio Model: {MODEL_NAME}")
    from openai import OpenAI
    client = OpenAI(api_key=LMSTUDIO_API_KEY, base_url=LMSTUDIO_API_URL, timeout=60)

elif ENGINE == "anything":
    ANYTHING_API_KEY = os.getenv("ANYTHING_API_KEY", "local")
    ANYTHING_API_URL = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
    MODEL_NAME = os.getenv("MODEL_NAME_ANY", os.getenv("MODEL_NAME", "llama-3.3-70b-instruct"))
    print(f"[DEBUG] Anything URL: {ANYTHING_API_URL}")
    print(f"[DEBUG] Anything Model: {MODEL_NAME}")
    from openai import OpenAI
    client = OpenAI(api_key=ANYTHING_API_KEY, base_url=ANYTHING_API_URL, timeout=60)

elif ENGINE == "claude":
    # Bezpośrednia integracja Claude (jak w zad1.py)
    try:
        from anthropic import Anthropic
    except ImportError:
        print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not CLAUDE_API_KEY:
        print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env", file=sys.stderr)
        sys.exit(1)
    
    MODEL_NAME = os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    claude_client = Anthropic(api_key=CLAUDE_API_KEY)

elif ENGINE == "gemini":
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("❌ Brak GEMINI_API_KEY w .env lub zmiennych środowiskowych.", file=sys.stderr)
        sys.exit(1)
    MODEL_NAME = os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel(MODEL_NAME)
else:
    print("❌ Nieobsługiwany silnik:", ENGINE, file=sys.stderr)
    sys.exit(1)

# ── 6. odpowiedzi ────────────────────────────────────────────────────────────
def answer_with_llm(question: str) -> str:
    lang = detect_lang(question)
    if ENGINE in {"openai", "lmstudio", "anything"}:
        try:
            print(f"[DEBUG] Wysyłam zapytanie do {ENGINE}: {question}")
            rsp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[lang]},
                    {"role": "user",   "content": question},
                ],
                max_tokens=10,
                temperature=0,
            )
            answer = rsp.choices[0].message.content.strip()
            print(f"[DEBUG] Otrzymana odpowiedź: {answer}")
            
            # Liczenie tokenów (już było w oryginalnym kodzie)
            tokens = rsp.usage
            print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
            if ENGINE == "openai":
                cost = tokens.prompt_tokens/1_000_000*0.60 + tokens.completion_tokens/1_000_000*2.40
                print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
            elif ENGINE in {"lmstudio", "anything"}:
                print(f"[💰 Model lokalny ({ENGINE}) - brak kosztów]")
            return answer
        except Exception as e:
            print("[!] LLM error:", e, file=sys.stderr)
            return {
                "pl": "Nie wiem",
                "fr": "Je ne sais pas",
                "en": "I don't know",
            }[lang]
    
    elif ENGINE == "claude":
        try:
            # Claude - bezpośrednia integracja
            resp = claude_client.messages.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": SYSTEM_PROMPTS[lang] + "\n\n" + question}],
                temperature=0,
                max_tokens=10
            )
            answer = resp.content[0].text.strip()
            
            # Liczenie tokenów Claude
            usage = resp.usage
            cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
            print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
            print(f"[💰 Koszt Claude: {cost:.6f} USD]")
            
            return answer
        except Exception as e:
            print("[!] Claude error:", e, file=sys.stderr)
            return {
                "pl": "Nie wiem",
                "fr": "Je ne sais pas", 
                "en": "I don't know",
            }[lang]
    
    elif ENGINE == "gemini":
        response = model_gemini.generate_content(
            [SYSTEM_PROMPTS[lang], question],
            generation_config={"temperature": 0.0, "max_output_tokens": 10}
        )
        print(f"[📊 Gemini - brak szczegółów tokenów]")
        print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
        return response.text.strip()

def decide_answer(question: str) -> str:
    return answer_locally(question) or answer_with_llm(question)

# ── 7. pętla rozmowy z serwerem ──────────────────────────────────────────────
def converse() -> None:
    print(f"🔄 Engine: {ENGINE}")
    session          = requests.Session()
    session.verify   = False
    msg_id           = 0
    outgoing         = {"text": "READY", "msgID": str(msg_id)}
    print(">>>", outgoing)

    while True:
        try:
            r = session.post(VERIFY_URL, json=outgoing,
                             auth=(USERNAME, PASSWORD),
                             headers=HDRS, timeout=10)
            r.raise_for_status()
            reply = r.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print("[!] HTTP/JSON:", e, file=sys.stderr); sys.exit(1)

        print("<<<", reply)
        text   = reply.get("text", "")
        msg_id = reply.get("msgID", msg_id)

        if "OK" in text:
            print("[✓] Uznani za androida.")
            return
        if "{{FLG:" in text:
            print("[★] Flaga:", text)
            return

        outgoing = {"text": decide_answer(text), "msgID": str(msg_id)}
        print(">>>", outgoing)
        time.sleep(0.3)

# ── 8. uruchom ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        converse()
    except KeyboardInterrupt:
        print("\n[!] Przerwane.")