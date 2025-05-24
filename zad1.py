#!/usr/bin/env python3
"""
robot_login.py  (multi-engine + Claude)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automatyzuje logowanie do systemu robotów z pytaniem anty-captcha
i rozwiązuje je za pomocą wybranego silnika LLM
(OpenAI, LM Studio, LocalAI / Anything LLM, Gemini Google, Claude Anthropic).
DODANO: Obsługę Claude z kompatybilnym interfejsem
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import Claude integration
try:
    from claude_integration import setup_claude_for_task, add_token_counting_to_openai_call
except ImportError:
    print("❌ Brak claude_integration.py - skopiuj plik z artefaktu")
    # Kontynuujemy bez Claude
    pass

# ── 0. CLI / env - wybór silnika ──────────────────────────────────────────────
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Login + captcha solver (multi-engine + Claude)")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
args = parser.parse_args()

ENGINE = (args.engine or os.getenv("LLM_ENGINE", "openai")).lower()

# ── 1. Dane logowania / stałe ────────────────────────────────────────────────
LOGIN_URL = os.getenv("ROBOT_LOGIN_URL")
USERNAME  = os.getenv("ROBOT_USERNAME")
PASSWORD  = os.getenv("ROBOT_PASSWORD")

ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"

SYSTEM_PROMPT = "Odpowiedz krótko: sama liczba / jedno słowo, bez wyjaśnień."

# ── 2. Inicjalizacja klienta LLM ─────────────────────────────────────────────
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
    from openai import OpenAI
    client = OpenAI(api_key=LMSTUDIO_API_KEY, base_url=LMSTUDIO_API_URL)

elif ENGINE == "anything":
    ANYTHING_API_KEY = os.getenv("ANYTHING_API_KEY", "local")
    ANYTHING_API_URL = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
    MODEL_NAME = os.getenv("MODEL_NAME_ANY", os.getenv("MODEL_NAME", "llama-3.3-70b-instruct"))
    from openai import OpenAI
    client = OpenAI(api_key=ANYTHING_API_KEY, base_url=ANYTHING_API_URL)

elif ENGINE == "claude":
    # Inicjalizacja Claude
    try:
        claude_client, MODEL_NAME = setup_claude_for_task(ENGINE)
        client = claude_client
    except:
        print("❌ Błąd inicjalizacji Claude - sprawdź claude_integration.py i CLAUDE_API_KEY", file=sys.stderr)
        sys.exit(1)

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

# ── 3. Pobranie pytania captcha ──────────────────────────────────────────────
def banner(title: str) -> str:
    if sys.stdout.isatty():
        return f"{ANSI_GREEN}=== {title} ==={ANSI_RESET}"
    return f"=== {title} ==="

def get_question(session: requests.Session, url: str) -> str:
    html = session.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    div  = soup.find(id="human-question")
    if div:
        return div.get_text(strip=True)
    m = re.search(r'<div[^>]*id=["\']human-question["\'][^>]*>([^<]+)</div>', html)
    if m:
        return m.group(1).strip()
    raise ValueError("Nie znaleziono pytania anty-captcha.")

# ── 4. Zapytanie LLM ─────────────────────────────────────────────────────────
def ask_llm(question: str) -> str:
    if ENGINE in {"openai", "lmstudio", "anything"}:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        
        # Liczenie tokenów (już było w oryginalnym kodzie)
        tokens = resp.usage
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        if ENGINE == "openai":
            cost = tokens.prompt_tokens/1_000_000*0.60 + tokens.completion_tokens/1_000_000*2.40
            print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        elif ENGINE in {"lmstudio", "anything"}:
            print(f"[💰 Model lokalny - brak kosztów]")
        
        m = re.search(r"(\d{1,4})", raw)
        return m.group(1) if m else raw
    
    elif ENGINE == "claude":
        resp = client.chat_completions_create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"(\d{1,4})", raw)
        return m.group(1) if m else raw
    
    elif ENGINE == "gemini":
        response = model_gemini.generate_content(
            [SYSTEM_PROMPT, question],
            generation_config={"temperature": 0.0, "max_output_tokens": 16}
        )
        raw = response.text.strip()
        print(f"[📊 Gemini - brak szczegółów tokenów]")
        print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
        m = re.search(r"(\d{1,4})", raw)
        return m.group(1) if m else raw

# ── 5. Logowanie i pobranie strony sekretu ───────────────────────────────────
def login_and_get_secret(sess: requests.Session, answer: str):
    data    = {"username": USERNAME, "password": PASSWORD, "answer": answer}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = sess.post(LOGIN_URL, data=data, headers=headers, allow_redirects=True)
    r.raise_for_status()
    url = r.url
    if url.rstrip("/") == LOGIN_URL.rstrip("/"):
        soup = BeautifulSoup(r.text, "html.parser")
        a    = soup.find("a", href=re.compile(r"secret", re.I))
        if not a:
            raise ValueError("Nie znaleziono linku do sekretnej strony.")
        url  = urljoin(LOGIN_URL, a["href"])
        r    = sess.get(url)
        r.raise_for_status()
    return url, r.text

# ── 6. Ekstrakcja flagi i plików .txt ────────────────────────────────────────
def extract_flag(html: str) -> str:
    for pat in (r"\{\{FLG:[^}]+\}\}", r"FLG\{[^}]+\}"):
        m = re.search(pat, html)
        if m:
            return m.group(0)
    raise ValueError("Nie znaleziono flagi.")

def fetch_txt_files(sess: requests.Session, secret_html: str):
    soup = BeautifulSoup(secret_html, "html.parser")
    hrefs = {a["href"] for a in soup.find_all("a", href=True)
             if a["href"].startswith("/files/") and a["href"].lower().endswith(".txt")}
    for dt in soup.find_all("dt"):
        m = re.search(r"Version (\d+\.\d+\.\d+)", dt.get_text(strip=True))
        if m:
            ver = m.group(1).replace(".", "_")
            hrefs.add(f"/files/{ver}.txt")
    for href in sorted(hrefs):
        url = urljoin(LOGIN_URL, href)
        try:
            resp = sess.get(url)
            resp.raise_for_status()
            yield url, resp.text
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                print(f"[!] Brak pliku {url} (404)", file=sys.stderr)
            else:
                raise

# ── 7. Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"🔄 Engine: {ENGINE}")
    with requests.Session() as s:
        print(banner("Pytanie captcha"))
        q = get_question(s, LOGIN_URL)
        print(q)
        print(banner("Odpowiedź LLM"))
        a = ask_llm(q)
        print(a)
        print(banner("Logowanie"))
        url, html = login_and_get_secret(s, a)
        print(f"URL sekretu: {url}")
        print(banner("Pełny HTML"))
        print(html)
        print(banner("Koniec HTML"))
        print(banner("Flaga"))
        print(extract_flag(html))
        print(banner("Pliki .txt"))
        for file_url, content in fetch_txt_files(s, html):
            print(f"\n-- {file_url} --\n{content}\n")

# ── 8. Uruchom ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[BŁĄD] {err}", file=sys.stderr)
        sys.exit(1)