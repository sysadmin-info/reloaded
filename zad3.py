#!/usr/bin/env python3
"""
json_fix_submit.py  (wersja z konfiguracją wyłącznie z pliku .env)

Pobiera plik kalibracyjny → poprawia → uzupełnia brakujące odpowiedzi przy pomocy LLM → wysyła wynik do Centrali.
Konfiguracja: wszystkie zmienne czytane z .env (bez parametrów CLI).
DODANO: Obsługę Claude + liczenie tokenów i kosztów dla wszystkich silników (bezpośrednia integracja)
"""

from __future__ import annotations
import os
import sys
import json
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ── 0. Wczytanie konfiguracji (env / .env) ───────────────────────────────────
load_dotenv(override=True)

# ── 1. Wybór silnika LLM wyłącznie z .env ────────────────────────────────────
ENGINE = os.getenv("LLM_ENGINE", "openai").lower()
if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
    sys.exit(1)

print(f"🔄 ENGINE: {ENGINE}")

# ── 2. Konfiguracja Centrali ────────────────────────────────────────────────
CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
if not CENTRALA_API_KEY:
    print("❌ Brak ustawionej zmiennej CENTRALA_API_KEY", file=sys.stderr)
    sys.exit(1)

REPORT_URL = os.getenv("REPORT_URL")
if not REPORT_URL:
    print("❌ Brak REPORT_URL w .env", file=sys.stderr)
    sys.exit(1)

SOURCE_URL = os.getenv("SOURCE_URL")
if not SOURCE_URL:
    print("❌ Brak SOURCE_URL w .env", file=sys.stderr)
    sys.exit(1)

SAVE_FILE = Path(os.getenv("SAVE_FILE", "poprawiony_json.json"))

# ── 3. Inicjalizacja klienta LLM ─────────────────────────────────────────────
if ENGINE == "openai":
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)

elif ENGINE in {"lmstudio", "anything"}:
    from openai import OpenAI
    api_key = os.getenv(f"{ENGINE.upper()}_API_KEY", "local")
    api_url = os.getenv(f"{ENGINE.upper()}_API_URL", "http://localhost:1234/v1")
    MODEL_NAME = os.getenv(f"MODEL_NAME_{ENGINE.upper()}", os.getenv("MODEL_NAME", "llama-3.3-70b-instruct"))
    client = OpenAI(api_key=api_key, base_url=api_url)

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
    
    MODEL_NAME = os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    claude_client = Anthropic(api_key=CLAUDE_API_KEY)

elif ENGINE == "gemini":
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("❌ Brak GEMINI_API_KEY w .env", file=sys.stderr)
        sys.exit(1)
    MODEL_NAME = os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel(MODEL_NAME)

# ── 4. Pobranie JSON‑a z Centrali ────────────────────────────────────────────
def download_json(url: str) -> dict[str, Any]:
    print("⬇️  Pobieram plik kalibracyjny…")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()

# ── 5. Prosta arytmetyka w treści pytania ────────────────────────────────────
_ARITH = re.compile(r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*$")

def eval_simple_expr(expr: str) -> int | None:
    m = _ARITH.match(expr)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    try:
        return {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
    except Exception:
        return None

# ── 6. LLM - hurtowe odpowiadanie ────────────────────────────────────────────
PROMPT_TMPL = (
    "Odpowiedz krótko na każde pytanie. "
    "Zwróć JSON - listę odpowiedzi w kolejności pytań. "
    "Jeśli nie wiesz, zwróć null.\nPytania:\n{qs}\n"
)

def answer_batch(batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    
    qs = [item["q"] for item in batch]
    prompt = PROMPT_TMPL.format(qs=json.dumps(qs, ensure_ascii=False))
    
    if ENGINE in {"openai", "lmstudio", "anything"}:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip() if response.choices else ""
        
        # Liczenie tokenów (jak w zad1.py i zad2.py)
        tokens = response.usage
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        if ENGINE == "openai":
            cost = tokens.prompt_tokens/1_000_000*0.60 + tokens.completion_tokens/1_000_000*2.40
            print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        elif ENGINE in {"lmstudio", "anything"}:
            print(f"[💰 Model lokalny - brak kosztów]")
        
    elif ENGINE == "claude":
        # Claude - bezpośrednia integracja (jak w zad1.py i zad2.py)
        response = claude_client.messages.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4000
        )
        raw = response.content[0].text.strip()
        
        # Liczenie tokenów Claude (jak w zad1.py i zad2.py)
        usage = response.usage
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015  # Claude Sonnet 4 pricing
        print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")
        
    elif ENGINE == "gemini":
        response = model_gemini.generate_content(
            [prompt],
            generation_config={"temperature": 0.0, "max_output_tokens": 512}
        )
        raw = response.text.strip()
        print(f"[📊 Gemini - brak szczegółów tokenów]")
        print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
    
    # Przetwarzanie odpowiedzi
    raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        answers = json.loads(raw)
        if not isinstance(answers, list):
            raise ValueError
    except Exception:
        found = re.search(r"\[.*?\]", raw, flags=re.S)
        answers = json.loads(found.group(0)) if found else []
    
    answers += [None] * max(0, len(batch) - len(answers))
    for rec, ans in zip(batch, answers[: len(batch)]):
        rec["a"] = ans

# ── 7. Transformacja danych (krok po kroku) ─────────────────────────────────
def fix_data(data: dict[str, Any]) -> dict[str, Any]:
    print("⚙️  Naprawiam dane…")
    data["apikey"] = CENTRALA_API_KEY
    raw_td = data.get("test-data", [])
    td = []
    if isinstance(raw_td, dict):
        td = [v for v in raw_td.values() if isinstance(v, dict)]
    elif isinstance(raw_td, list):
        td = raw_td
    
    batch: list[dict[str, Any]] = []
    
    for rec in td:
        if q := rec.get("question"):
            if result := eval_simple_expr(q):
                rec["answer"] = result
        if test := rec.get("test"):
            batch.append(test)
            if len(batch) >= 90:
                answer_batch(batch)
                batch.clear()
    
    answer_batch(batch)
    data["test-data"] = td
    return data

# ── 8. Wysłanie raportu do Centrali ─────────────────────────────────────────
def submit_report(answer: dict[str, Any]) -> None:
    print("📡 Wysyłam raport…")
    payload = {"task": "JSON", "apikey": CENTRALA_API_KEY, "answer": answer}
    resp = requests.post(REPORT_URL, json=payload, timeout=60)
    if resp.ok:
        print("🎉 Sukces! Odpowiedź serwera:", resp.json())
    else:
        print(f"❌ Błąd HTTP {resp.status_code}: {resp.text}", file=sys.stderr)

# ── 9. Główna logika ────────────────────────────────────────────────────────
def main() -> None:
    original = download_json(SOURCE_URL)
    fixed = fix_data(original)
    SAVE_FILE.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    print("💾 Zapisano lokalnie →", SAVE_FILE)
    submit_report(fixed)

if __name__ == "__main__":
    main()