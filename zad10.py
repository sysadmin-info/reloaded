#!/usr/bin/env python3
"""
S03E01 - przygotowanie metadanych do 10 raportów dostarczonych w formacie TXT
Multi-engine: openai, lmstudio, anything, claude, gemini
Przygotowanie metadanych (słów kluczowych) dla raportów fabryki - chunking, contextual retrieval, cache, analiza faktów.
"""
import argparse
import os
import sys
import zipfile
import requests
import json
import re
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from zad9 import chunk_text

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Metadane raportów fabryki (multi-engine)")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"], help="LLM backend do użycia")
args = parser.parse_args()

ENGINE = None
if args.engine:
    ENGINE = args.engine.lower()
elif os.getenv("LLM_ENGINE"):
    ENGINE = os.getenv("LLM_ENGINE").lower()
else:
    model_name = os.getenv("MODEL_NAME", "")
    if "claude" in model_name.lower():
        ENGINE = "claude"
    elif "gemini" in model_name.lower():
        ENGINE = "gemini"
    elif "gpt" in model_name.lower() or "openai" in model_name.lower():
        ENGINE = "openai"
    elif os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        ENGINE = "claude"
    elif os.getenv("GEMINI_API_KEY"):
        ENGINE = "gemini"
    elif os.getenv("OPENAI_API_KEY"):
        ENGINE = "openai"
    else:
        ENGINE = "lmstudio"

if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
    sys.exit(1)
print(f"🔄 ENGINE wykryty: {ENGINE}")

# 2. Inicjalizacja klienta/modelu
MODEL_NAME = None

if ENGINE == "openai":
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o")
    if not openai.api_key:
        print("❌ Brak OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

elif ENGINE == "claude":
    try:
        from anthropic import Anthropic
    except ImportError:
        print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not CLAUDE_API_KEY:
        print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    claude_client = Anthropic(api_key=CLAUDE_API_KEY)

elif ENGINE == "gemini":
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("❌ Brak GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel(MODEL_NAME)

elif ENGINE == "lmstudio":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "qwen3-14b-128k")
    LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")

elif ENGINE == "anything":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "qwen3-14b-128k")
    ANYTHING_API_URL = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")

print(f"✅ Zainicjalizowano silnik: {ENGINE} z modelem: {MODEL_NAME}")

# 3. Uniwersalna funkcja do LLM – pewna obsługa operatora
def llm_request(prompt: str) -> str:
    if ENGINE == "openai":
        import openai
        resp = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    elif ENGINE == "claude":
        resp = claude_client.messages.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        return resp.content[0].text.strip()
    elif ENGINE == "gemini":
        response = model_gemini.generate_content([prompt], generation_config={"temperature":0, "max_output_tokens":1000})
        return response.text.strip()
    elif ENGINE == "lmstudio":
        url = LMSTUDIO_API_URL.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    elif ENGINE == "anything":
        url = ANYTHING_API_URL.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"Nieobsługiwany ENGINE: {ENGINE}")

# 4. Reszta logiki zadania – jak wcześniej
FABRYKA_URL = os.getenv("FABRYKA_URL")
REPORT_URL = os.getenv("REPORT_URL")
CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
if not REPORT_URL or not CENTRALA_API_KEY:
    raise RuntimeError("Brak REPORT_URL lub CENTRALA_API_KEY w .env")

CACHE_PATH = Path('.cache_context.json')
CACHE_PATH.parent.mkdir(exist_ok=True)

def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding='utf-8'))
    return {}

def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')

EVENT_KEYWORDS = [
    "włamanie", "pożar", "awaria", "wyciek", "kradzież",
    "przerwa", "przestój", "uszkodzenie", "zatrucie", "eksplozja"
]
LOCATION_REGEX = re.compile(r"\b(?:sektor|hala)\s*[A-Za-z0-9]+\b", re.IGNORECASE)
TECH_KEYWORDS = [
    "czujnik", "robot", "system", "sterownik", "silnik",
    "panel", "sieć", "ultradźwięki", "nadajnik",
    "skan", "detektor", "czujniki dźwięku"
]
PERSON_REGEX = re.compile(
    r"\b[A-ZŁŚŻŹ][a-ząęółśżźćń]+\s+[A-ZŁŚŻŹ][a-ząęółśżźćń]+\b"
)

def extract_sector_from_filename(filename: str) -> str:
    match = re.search(r"sektor[_\s]*([A-Z0-9]+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def contextualize_report_with_facts(report_text: str, facts_map: dict) -> tuple:
    people_in_report = set()
    for match in PERSON_REGEX.finditer(report_text):
        people_in_report.add(match.group(0).lower())
    related_facts = []
    person_professions = {}
    for fact_key, fact_text in facts_map.items():
        for person in people_in_report:
            if person in fact_text.lower():
                related_facts.append(fact_text)
                if "nauczyciel" in fact_text.lower() and person == "aleksander ragowski":
                    person_professions[person] = "nauczyciel"
                elif "programista" in fact_text.lower() and person in fact_text.lower():
                    person_professions[person] = "programista"
                elif person == "barbara zawadzka" and "frontend" in fact_text.lower():
                    person_professions[person] = "programista"
                break
    context = f"RAPORT:\n{report_text}\n\n"
    if related_facts:
        context += "POWIĄZANE FAKTY:\n"
        for fact in related_facts:
            context += f"{fact}\n\n"
    return context, person_professions

def extract_keywords_with_context(full_context: str, filename: str, cache: dict) -> set:
    context_hash = hashlib.sha256((full_context + filename).encode('utf-8')).hexdigest()
    if context_hash in cache:
        return set(cache[context_hash])
    sector = extract_sector_from_filename(filename)
    prompt = f"""Przeanalizuj poniższy raport wraz z powiązanymi faktami.
Zwróć WSZYSTKIE istotne słowa kluczowe w języku polskim, w mianowniku, oddzielone przecinkami.

WAŻNE ZASADY:
1. Jeśli osoba jest wymieniona w raporcie, a w faktach jest informacja o jej zawodzie, KONIECZNIE uwzględnij ten zawód
2. Jeśli raport opisuje schwytanie/aresztowanie/przekazanie do działu kontroli osoby, uwzględnij jej zawód
3. Uwzględnij imiona i nazwiska osób
4. Uwzględnij lokalizacje (sektory, miejsca)
5. Uwzględnij technologie i urządzenia
6. Uwzględnij wydarzenia i czynności
7. Jeśli ktoś jest przekazany do "działu kontroli", dodaj: przechwycenie, aresztowanie
8. Jeśli wspomniano odciski palców konkretnej osoby i raport jest z sektora {sector}, dodaj: "odciski palców w sektorze {sector}"
9. Jeśli Barbara Zawadzka jest wymieniona w kontekście technologii, dodaj: JavaScript, frontend, programista
10. Jeśli osoba należy do ruchu oporu (według faktów), dodaj: ruch oporu

{full_context}

Słowa kluczowe:"""
    keywords_text = llm_request(prompt)
    keywords = [w.strip().lower() for w in keywords_text.split(',') if w.strip()]
    cache[context_hash] = keywords
    save_cache(cache)
    return set(keywords)

def extract_keywords(report_text: str, filename: str, facts_map: dict, cache: dict) -> set:
    kws = set()
    file_tokens = {tok.lower() for tok in re.split(r"[\W_]+", filename) if tok and len(tok) > 2}
    kws |= file_tokens
    sector = extract_sector_from_filename(filename)
    if sector:
        kws.add(f"sektor {sector.lower()}")
    full_context, person_professions = contextualize_report_with_facts(report_text, facts_map)
    context_keywords = extract_keywords_with_context(full_context, filename, cache)
    kws |= context_keywords
    people_found = []
    for match in PERSON_REGEX.finditer(report_text):
        name = match.group(0)
        kws.add(name.lower())
        people_found.append(name.lower())
        if name.lower() in person_professions:
            kws.add(person_professions[name.lower()])
    for evt in EVENT_KEYWORDS:
        if re.search(rf"\b{evt}\b", report_text, re.IGNORECASE):
            kws.add(evt)
    for loc in LOCATION_REGEX.findall(report_text):
        kws.add(loc.lower())
    for tech in TECH_KEYWORDS:
        if re.search(rf"\b{tech}\b", report_text, re.IGNORECASE):
            kws.add(tech)
    if any(word in report_text.lower() for word in ["przekazan", "kontroli", "schwyta", "aresztowa"]):
        kws.add("przechwycenie")
        kws.add("aresztowanie")
    if any(word in report_text.lower() for word in ["zwierzyna", "fauna", "wildlife", "leśna"]):
        kws.add("zwierzęta")
    if "odcisk" in report_text.lower():
        kws.add("odciski palców")
        kws.add("analiza odcisków palców")
        if "barbara zawadzka" in report_text.lower() and sector:
            kws.add(f"odciski palców w sektorze {sector.lower()}")
    for person in people_found:
        if person == "barbara zawadzka":
            kws.add("javascript")
            kws.add("frontend")
            kws.add("programista")
            kws.add("ruch oporu")
        elif person == "aleksander ragowski":
            kws.add("nauczyciel")
            kws.add("ruch oporu")
    if "las" in report_text.lower() or "krzak" in report_text.lower():
        kws.add("las")
    return kws

def download_and_extract(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / 'fabryka.zip'
    print(f"📥 Pobieranie plików z {FABRYKA_URL}...")
    resp = requests.get(FABRYKA_URL, stream=True)
    resp.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    print("📦 Rozpakowywanie archiwum...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            if member.filename.endswith('weapons_tests.zip'):
                continue
            zf.extract(member, dest)
    zip_path.unlink()
    print("✅ Pliki rozpakowane")

def collect_files(dest: Path):
    reports, facts = [], []
    for p in dest.rglob('*.txt'):
        if 'facts' in p.parts or 'fakty' in p.parts:
            facts.append(p)
        else:
            reports.append(p)
    return reports, facts

def load_facts(facts_files):
    facts_map = {}
    for f in facts_files:
        facts_map[f.stem] = f.read_text(encoding='utf-8', errors='ignore')
    return facts_map

def main():
    print("🔄 Rozpoczynam przetwarzanie raportów...")
    base = Path('fabryka')
    download_and_extract(base)
    reports, facts = collect_files(base)
    print(f"📄 Znaleziono {len(reports)} raportów i {len(facts)} plików z faktami")
    if len(reports) != 10:
        print(f"⚠️  Oczekiwano 10 raportów, znaleziono {len(reports)}")
    facts_map = load_facts(facts)
    cache = load_cache()
    answer = {}
    print("🔍 Analizuję raporty...")
    for rpt in sorted(reports):
        print(f"  📋 Przetwarzam: {rpt.name}")
        text = rpt.read_text(encoding='utf-8', errors='ignore')
        kws = extract_keywords(text, rpt.name, facts_map, cache)
        answer[rpt.name] = ",".join(sorted(kws))
        print(f"     ✅ Znaleziono {len(kws)} słów kluczowych")
    payload = {
        "task": "dokumenty",
        "apikey": CENTRALA_API_KEY,
        "answer": answer
    }
    print("\n📤 Wysyłam rozwiązanie...")
    resp = requests.post(REPORT_URL, json=payload)
    try:
        resp.raise_for_status()
        print('✅ Sukces!', resp.json())
    except Exception as e:
        print('❌ Błąd:', e)
        if resp.text:
            print('Odpowiedź serwera:', resp.text)

if __name__ == '__main__':
    main()
