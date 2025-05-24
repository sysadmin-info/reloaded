#!/usr/bin/env python3
"""
zad9.py - Klasyfikacja plików z fabryki: jedna prośba do LLM na plik, detekcja języka, zwracanie kategorii
• Multiengine: openai / gemini / lmstudio / anything / claude
• Ekstrakcja: txt→tekst, mp3/wav→Whisper lokalnie, png/jpg→OCR (OpenCV+pytesseract)
• Orkiestracja: LangGraph

POPRAWKI: Konserwatywna logika klasyfikacji people - tylko potwierdzone schwytania
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

print(f"🔄 Engine: {ENGINE}")

# wybór modelu z .env
if ENGINE == "gemini":
    MODEL_NAME = os.getenv("MODEL_NAME_GEMINI")
elif ENGINE == "lmstudio":
    MODEL_NAME = os.getenv("MODEL_NAME_LM")
elif ENGINE == "anything":
    MODEL_NAME = os.getenv("MODEL_NAME_ANY")
elif ENGINE == "claude":
    MODEL_NAME = os.getenv("MODEL_NAME_CLAUDE")
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

elif ENGINE == 'claude':
    # Bezpośrednia integracja Claude
    try:
        from anthropic import Anthropic
    except ImportError:
        print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not CLAUDE_API_KEY:
        print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env", file=sys.stderr)
        sys.exit(1)
    
    claude_client = Anthropic(api_key=CLAUDE_API_KEY)

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
        # Liczenie tokenów
        tokens = resp.usage
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        cost = tokens.prompt_tokens/1_000_000*0.60 + tokens.completion_tokens/1_000_000*2.40
        print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        return resp.choices[0].message.content.strip().lower()
    
    if ENGINE == 'claude':
        # Claude - bezpośrednia integracja
        resp = claude_client.messages.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4000
        )
        
        # Liczenie tokenów Claude
        usage = resp.usage
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015  # Claude Sonnet 4 pricing
        print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")
        
        return resp.content[0].text.strip().lower()
    
    if ENGINE == 'gemini':
        response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        print(f"[📊 Gemini - brak szczegółów tokenów]")
        print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
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
        print(f"[📊 LMStudio - brak szczegółów tokenów]")
        print(f"[💰 LMStudio - model lokalny, brak kosztów]")
        return content.strip().lower()
    
    if ENGINE == 'anything':
        headers = {"Authorization": f"Bearer {ANYTHING_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "inputs": prompt}
        resp = requests.post(ANYTHING_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        print(f"[📊 Anything - brak szczegółów tokenów]")
        print(f"[💰 Anything - model lokalny, brak kosztów]")
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

# --- 6. Klasyfikacja pliku (ORYGINALNA LOGIKA ZE STAREJ WERSJI) ---
def classify_file(text: str, filename: str) -> str:
    lang = detect_language(text)
    low = text.lower()
    
    # Debug - pokaż fragment tekstu dla analizy
    print(f"[DEBUG] Text fragment: {low[:200]}...")
    
    # Heuristic shortcuts for OpenAI - dodane żeby łapać nadajniki z odciskami
    if ENGINE == 'openai':
        # Nadajnik + odciski palców = definitywnie people
        if ('nadajnik' in low or 'transmitter' in low) and ('odcisk' in low or 'fingerprint' in low):
            print(f"[HEURISTIC] People detected: transmitter with fingerprints")
            return 'people'
        # Podstawowe słowa dla people
        presence_kw = ['found one guy', 'captured', 'schwytanych', 'wykryto jednostkę organiczną', 'przedstawił się jako']
        if any(kw in low for kw in presence_kw):
            print(f"[HEURISTIC] People detected: {[kw for kw in presence_kw if kw in low]}")
            return 'people'
    
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
    
    # OpenAI/Gemini prompt (ORYGINALNA WERSJA)
    if ENGINE in ('openai', 'gemini', 'claude'):
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
    # LMStudio/Anything detailed few-shot (ORYGINALNA WERSJA)
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
    
    # Walidacja odpowiedzi
    if cat in {'people', 'hardware', 'other'}:
        return cat
    else:
        print(f"[WARNING] Nieoczekiwana odpowiedź LLM: '{result}' -> defaulting to 'other'")
        return 'other'

# --- 7. Pipeline LangGraph ---
def download_node(state):
    """Node funkcja dla pobierania danych"""
    download_and_extract(Path('fabryka'))
    return state

def classify_node(state):
    """Node funkcja dla klasyfikacji plików"""
    root = Path('fabryka')
    files = [p for p in root.rglob('*') if p.is_file() and 'facts' not in p.parts and p.name != 'weapons_tests.zip']
    print(f"[CLASSIFY] Found {len(files)} files")
    cats = {'people':[], 'hardware':[], 'other':[]}
    
    for fp in sorted(files):
        print(f"\n[CLASSIFY] Processing: {fp.name}")
        
        # Ekstrakcja tekstu
        if fp.suffix == '.txt': 
            text = extract_text(fp)
        elif fp.suffix in ['.mp3','.wav']: 
            text = extract_audio(fp)
        elif fp.suffix in ['.png','.jpg','.jpeg']: 
            text = extract_image(fp)
        else: 
            text = ''
        
        # Debug snippet
        snippet = text.replace('\n',' ')[:100]
        print(f"[CLASSIFY] Snippet: {snippet!r}")
        
        # Klasyfikacja
        cat = classify_file(text, fp.name)
        print(f"[CLASSIFY] Result: {cat}")
        
        cats[cat].append(fp.name)
    
    # Zapis surowej klasyfikacji do debugowania
    Path('raw_classification.json').write_text(json.dumps(cats, ensure_ascii=False, indent=4), encoding='utf-8')
    
    # Zwracamy stan z wynikami klasyfikacji
    state.update(cats)
    return state

def aggregate_node(state):
    """Node funkcja dla agregacji wyników"""
    ppl = sorted(state.get('people',[]))
    hw = sorted(state.get('hardware',[]))
    out = {'people':ppl, 'hardware':hw}
    
    print(f"\n[AGGREGATE] Final results:")
    print(f"  People: {len(ppl)} files: {ppl}")
    print(f"  Hardware: {len(hw)} files: {hw}")
    print(f"  Other: {len(state.get('other', []))} files (not included in report)")
    
    Path('wynik.json').write_text(json.dumps(out, ensure_ascii=False, indent=4), encoding='utf-8')
    state['report'] = out
    return state

def send_node(state):
    """Node funkcja dla wysyłania raportu"""
    payload = {'task':'kategorie', 'apikey':CENTRALA_API_KEY, 'answer':state.get('report')}
    Path('payload.json').write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding='utf-8')
    
    print(f"\n[SEND] Sending payload: {payload}")
    resp = requests.post(REPORT_URL, json=payload)
    print(f"[SEND] Centralna odpowiedź: {resp.text}")
    return state

def build_graph():
    """Buduje graf LangGraph z właściwymi funkcjami"""
    graph = StateGraph(input=dict, output=dict)
    
    # Dodawanie node'ów
    graph.add_node('download', download_node)
    graph.add_edge(START, 'download')
    
    graph.add_node('classify', classify_node)
    graph.add_edge('download','classify')
    
    graph.add_node('aggregate', aggregate_node)
    graph.add_edge('classify','aggregate')
    
    graph.add_node('send', send_node)
    graph.add_edge('aggregate','send')
    graph.add_edge('send', END)
    
    return graph.compile()

# --- 8. main ---
def main():
    print("=== Zadanie 9: klasyfikacja plików z fabryki ===")
    print("LOGIKA: Oryginalna + fix dla OpenAI (nadajnik z odciskami = people)")
    print("OCZEKIWANE: people=3, hardware=3 pliki")
    print("Startuje pipeline...\n")
    build_graph().invoke({})

if __name__=='__main__': main()