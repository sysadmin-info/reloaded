#!/usr/bin/env python3
"""
zad9.py - Klasyfikacja plików z fabryki: jedna prośba do LLM na plik, detekcja języka, zwracanie kategorii
• Multiengine: openai / gemini / lmstudio / anything
• Ekstrakcja: txt→tekst, mp3/wav→Whisper lokalnie, png/jpg→OCR (OpenCV+pytesseract)
• Orkiestracja: LangGraph

Logika: dla każdego pliku:
 - ekstrakcja → pełny tekst
 - detekcja języka (pl/en)
 - prompt do LLM w odpowiednim języku → kategoria (people/hardware/other)
Debug pokazuje każdy krok, zapisuje surową i ostateczną klasyfikację oraz payload.
"""
import os
import sys
import json
import zipfile
from pathlib import Path
import requests
from dotenv import load_dotenv
import cv2
import pytesseract
import whisper
from langdetect import detect
from langgraph.graph import StateGraph, START, END

# --- 1. Konfiguracja i inicjalizacja LLM ---
load_dotenv(override=True)
CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
FABRYKA_URL     = os.getenv("FABRYKA_URL")
REPORT_URL      = os.getenv("REPORT_URL")
ENGINE          = os.getenv("LLM_ENGINE", "openai").lower()

# wybór modelu z .env
if ENGINE == "gemini":
    MODEL_NAME = os.getenv("MODEL_NAME_GEMINI")
elif ENGINE == "lmstudio":
    MODEL_NAME = os.getenv("MODEL_NAME_LM")
elif ENGINE == "anything":
    MODEL_NAME = os.getenv("MODEL_NAME_ANY")
else:
    MODEL_NAME = os.getenv("MODEL_NAME_OPENAI")

# klucze i URL-e
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY")
LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL")
ANYTHING_API_KEY = os.getenv("ANYTHING_API_KEY")
ANYTHING_API_URL = os.getenv("ANYTHING_API_URL")

# weryfikacja
if not all([CENTRALA_API_KEY, FABRYKA_URL, REPORT_URL, MODEL_NAME]):
    print("❌ Ustaw w .env wszystkie: CENTRALA_API_KEY, FABRYKA_URL, REPORT_URL, MODEL_NAME_*")
    sys.exit(1)

# inicjalizacja klienta OpenAI i Gemini
if ENGINE == 'openai':
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))
elif ENGINE == 'gemini':
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- 2. Init Whisper ---
audio_model = whisper.load_model(os.getenv("WHISPER_MODEL", "small"))

# --- 3. Wywołanie LLM ---
def call_llm(prompt: str) -> str:
    if ENGINE == 'openai':
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip().lower()
    if ENGINE == 'gemini':
        response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        return response.text.strip().lower()
    if ENGINE == 'lmstudio':
        # LMStudio expects /chat/completions endpoint
        url = LMSTUDIO_API_URL.rstrip('/') + '/chat/completions'
        headers = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content.strip().lower()
    if ENGINE == 'anything':
        headers = {"Authorization": f"Bearer {ANYTHING_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "inputs": prompt}
        resp = requests.post(ANYTHING_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json().get('generated_text', '').strip().lower()
    raise ValueError(f"Nieobsługiwany silnik: {ENGINE}")

# --- 4. Ekstrakcja zawartości ---
def download_and_extract(dest: Path) -> None:
    if (dest / "2024-11-12_report-00-sektor_C4.txt").exists():
        print("[INFO] Pliki już rozpakowane - pomijam pobieranie.")
        return
    dest.mkdir(exist_ok=True)
    zip_path = dest / "fabryka.zip"
    print("[INFO] Pobieram dane z fabryki…")
    resp = requests.get(FABRYKA_URL, stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(8192): f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zf: zf.extractall(dest)
    zip_path.unlink()

def extract_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def extract_audio(fp: Path) -> str:
    result = audio_model.transcribe(str(fp)); text = result.get("text", "")
    Path("debug").mkdir(exist_ok=True)
    with open(f"debug/{fp.name}.txt", "w", encoding="utf-8") as f: f.write(text)
    return text

def extract_image(fp: Path) -> str:
    try:
        img = cv2.imread(str(fp)); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, lang='pol')
    except: return ''

# --- 5. Detekcja języka ---
def detect_language(text: str) -> str:
    try: lang = detect(text)
    except: lang = 'en'
    return 'pl' if lang.startswith('pl') else 'en'

# --- 6. Klasyfikacja pliku ---
def classify_file(text: str, filename: str) -> str:
    lang = detect_language(text)
    low = text.lower()
    # Heuristic shortcuts for LMStudio (local llama) to improve accuracy
    if ENGINE == 'lmstudio':
        hardware_kw = ['napraw', 'uster']
        if any(kw in low for kw in hardware_kw):
            return 'hardware'
        presence_kw = ['found one guy', 'captured', 'infiltrator', 'organiczna', 'schwytanych', 'ultradźwięk', 'osobnik', 'przechwyc']
        if any(kw in low for kw in presence_kw):
            return 'people'
    # Heuristic shortcuts for Gemini to catch obvious captures
    if ENGINE == 'gemini':
        # treat 'arrest' or 'found one guy' or 'captured' as people
        presence_kw = ['found one guy', 'captured', 'infiltrator', 'organiczna', 'schwytanych', 'ultradźwięk', 'osobnik', 'przechwyc']
        if any(kw in low for kw in presence_kw):
            return 'people'
        hardware_kw = ['napraw', 'uster']
        if any(kw in low for kw in hardware_kw):
            return 'hardware'
    # OpenAI/Gemini prompt
    if ENGINE in ('openai', 'gemini'):
        if lang == 'pl':
            prompt = f"""
Plik: {filename}
Zawartość:
{text}

Zadanie: przypisz do jednej z kategorii:
- people (informacje o schwytanych ludziach lub ich śladach obecności)
- hardware (naprawione usterki hardware'u)
- other (wszystko inne)

Odpowiedz tylko: people/hardware/other.
Upewnij się, że klasyfikujesz tylko wtedy, gdy są wyraźne informacje o schwytanych osobach.
Jeśli to tylko poszukiwania lub brak wyników - zaklasyfikuj jako 'other'.
"""
        else:
            prompt = f"""
File: {filename}
Content:
{text}

Task: classify into one category:
- people (notes about captured people or traces of their presence)
- hardware (fixed hardware malfunctions)
- other (anything else)

Answer only: people/hardware/other.
Only classify as 'people' if actual capture or presence is confirmed. 
Mere searches or absence should be classified as 'other'.
"""
    # LMStudio/Anything detailed few-shot
    else:
        examples = [
            {"filename": "sector_gate.mp3", "content": "We captured two infiltrators near the gate.", "category": "people"},
            {"filename": "antenna_repair.png", "content": "REPAIR NOTE: antenna module replaced at 07:30, hardware malfunction solved.", "category": "hardware"}
        ]
        if lang == 'pl':
            prompt = f"""
Plik: {filename}
Zawartość:
{text}

Przykłady:
- Plik: {examples[0]['filename']}, Zawartość: {examples[0]['content']}, Kategoria: {examples[0]['category']}
- Plik: {examples[1]['filename']}, Zawartość: {examples[1]['content']}, Kategoria: {examples[1]['category']}

Zadanie: przypisz do jednej z kategorii:
- people (informacje o schwytanych ludziach lub ich śladach obecności)
- hardware (naprawione usterki hardware'u)
- other (wszystko inne)

Odpowiedz jednym słowem: people, hardware lub other.
"""
        else:
            prompt = f"""
File: {filename}
Content:
{text}

Examples:
- File: {examples[0]['filename']}, Content: {examples[0]['content']}, Category: {examples[0]['category']}
- File: {examples[1]['filename']}, Content: {examples[1]['content']}, Category: {examples[1]['category']}

Task: classify into one of:
- people (notes about captured individuals or traces of their presence)
- hardware (fixed hardware malfunctions)
- other (anything else)

Answer one word: people, hardware or other.
"""
    result = call_llm(prompt)
    cat = result.strip().lower()
    return cat if cat in {'people','hardware','other'} else 'other'

# --- 7. Pipeline LangGraph --- Pipeline LangGraph --- Pipeline LangGraph --- Pipeline LangGraph --- Pipeline LangGraph --- Pipeline LangGraph ---
# (bez zmian od oryginału)
def build_graph():
    graph = StateGraph(input=dict, output=dict)
    graph.add_node('download', lambda st: download_and_extract(Path('fabryka')))
    graph.add_edge(START, 'download')
    def classify(st):
        root = Path('fabryka')
        files = [p for p in root.rglob('*') if p.is_file() and 'facts' not in p.parts and p.name != 'weapons_tests.zip']
        print(f"[CLASSIFY] Found {len(files)} files")
        cats = {'people':[], 'hardware':[], 'other':[]}
        for fp in sorted(files):
            if fp.suffix == '.txt': text = extract_text(fp)
            elif fp.suffix in ['.mp3','.wav']: text = extract_audio(fp)
            elif fp.suffix in ['.png','.jpg','.jpeg']: text = extract_image(fp)
            else: text = ''
            snippet = text.replace('\n',' ')[:100]
            print(f"[CLASSIFY] {fp.name} -> snippet: {snippet!r}")
            cat = classify_file(text, fp.name)
            print(f"[CLASSIFY] -> {cat}\n")
            cats[cat].append(fp.name)
        Path('raw_classification.json').write_text(json.dumps(cats, ensure_ascii=False, indent=4), encoding='utf-8')
        return cats
    graph.add_node('classify', classify)
    graph.add_edge('download','classify')
    def aggregate(st):
        ppl = sorted(st.get('people',[])); hw = sorted(st.get('hardware',[]))
        out = {'people':ppl, 'hardware':hw}
        Path('wynik.json').write_text(json.dumps(out, ensure_ascii=False, indent=4), encoding='utf-8')
        return {'report':out}
    graph.add_node('aggregate', aggregate)
    graph.add_edge('classify','aggregate')
    def send(st):
        payload = {'task':'kategorie', 'apikey':CENTRALA_API_KEY, 'answer':st.get('report')}
        Path('payload.json').write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding='utf-8')
        resp = requests.post(REPORT_URL, json=payload)
        print('Centralna odpowiedź:', resp.text)
        return {}
    graph.add_node('send', send)
    graph.add_edge('aggregate','send'); graph.add_edge('send', END)
    return graph.compile()

# --- 8. main ---
def main():
    print("Zadanie 9: klasyfikacja plików z fabryki...")
    build_graph().invoke({})

if __name__=='__main__': main()
