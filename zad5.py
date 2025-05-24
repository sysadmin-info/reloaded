#!/usr/bin/env python3
"""
zad5.py - Analiza nagrań audio z przesłuchań
Obsługuje: openai, lmstudio, anything, gemini, claude
DODANO: Obsługę Claude z bezpośrednią integracją (jak zad1.py i zad2.py)
"""
import os
import zipfile
import requests
import sys
from pathlib import Path
from dotenv import load_dotenv

# Konfiguracja i helpery
load_dotenv()
ENGINE = os.getenv("LLM_ENGINE", "openai").lower()

print(f"🔄 Engine: {ENGINE}")

DATA_URL       = os.getenv("DATA_URL")
REPORT_URL     = os.getenv("REPORT_URL")
CENTRALA_KEY   = os.getenv("CENTRALA_API_KEY")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME_GEMINI = os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")

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
        print("❌ Brak GEMINI_API_KEY w .env lub zmiennych środowiskowych.", file=sys.stderr)
        sys.exit(1)
    MODEL_NAME = os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel(MODEL_NAME)
else:
    print("❌ Nieobsługiwany silnik:", ENGINE, file=sys.stderr)
    sys.exit(1)

def download_and_extract_zip(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "przesluchania.zip"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)
    zip_path.unlink()

def find_audio_files(root: Path, exts=None) -> list[Path]:
    if exts is None:
        exts = ['.m4a', '.mp3', '.wav', '.flac']
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts)

def get_transcript(audio_path: Path) -> str:
    """Pobiera transkrypcję z cache lub generuje nową via Whisper i zapisuje lokalnie."""
    txt_path = audio_path.with_suffix('.txt')
    if txt_path.exists():
        print(f"   > Używam zapisanej transkrypcji: {txt_path.name}")
        return txt_path.read_text(encoding='utf-8')
    # brak cache, generujemy transkrypcję (tylko OpenAI Whisper)
    if ENGINE == "gemini":
        print("❌ Transkrypcja audio (Whisper) działa tylko przez OpenAI API.")
        exit(1)
    if ENGINE == "claude":
        print("❌ Transkrypcja audio (Whisper) wymaga OpenAI API - Claude nie obsługuje audio.")
        exit(1)
    print(f"   > Transkrypcja z chmury dla: {audio_path.name}")
    with open(audio_path, 'rb') as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text",
            language="pl"
        )
    text = getattr(resp, 'text', resp)
    txt_path.write_text(text, encoding='utf-8')
    return text

def infer_answer(fragments: str) -> str:
    system_msg = (
        "Jesteś śledczym-logiki."
        "Otrzymasz dwa fragmenty zeznań dotyczące przedmiotu i miejsca wykładów."
        "1. Wypisz \"Fakt 1\" - przedmiot wykładów."
        "2. Wypisz \"Fakt 2\" - miasto wykładów."
        "3. Na podstawie tych faktów wnioskuj nazwę wydziału uczelni."
        "4. Korzystając z wiedzy ogólnej, podaj ulicę siedziby wydziału."
        "Najpierw rozpisz chain-of-thought, potem odpowiedź w formacie:"
        "Wydział: <pełna nazwa>Ulica: <ulica i numer>"
    )
    user_msg = f"Fragmenty zeznań:{fragments}"
    
    if ENGINE in {"openai", "lmstudio", "anything"}:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0
        )
        answer = resp.choices[0].message.content.strip()
        
        # Liczenie tokenów (jak w zad1.py i zad2.py)
        tokens = resp.usage
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        if ENGINE == "openai":
            cost = tokens.prompt_tokens/1_000_000*0.60 + tokens.completion_tokens/1_000_000*2.40
            print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        elif ENGINE in {"lmstudio", "anything"}:
            print(f"[💰 Model lokalny - brak kosztów]")
        return answer
    
    elif ENGINE == "claude":
        # Claude - bezpośrednia integracja (jak w zad1.py i zad2.py)
        resp = claude_client.messages.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": system_msg + "\n\n" + user_msg}
            ],
            temperature=0,
            max_tokens=4000
        )
        
        # Liczenie tokenów Claude (jak w zad1.py i zad2.py)
        usage = resp.usage
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015  # Claude Sonnet 4 pricing
        print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")
        
        return resp.content[0].text.strip()
    
    elif ENGINE == "gemini":
        response = model_gemini.generate_content(
            [system_msg, user_msg],
            generation_config={"temperature": 0.0, "max_output_tokens": 512}
        )
        print(f"[📊 Gemini - brak szczegółów tokenów]")
        print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
        return response.text.strip()

def main():
    base_dir = Path("przesluchania")
    # 1. Pobierz i rozpakuj nagrania tylko jeśli nie ma plików audio
    audio_files = find_audio_files(base_dir) if base_dir.exists() else []
    if not audio_files:
        print("1/4 Pobieram i rozpakowuję nagrania...")
        download_and_extract_zip(DATA_URL, base_dir)
        audio_files = find_audio_files(base_dir)
    else:
        print("1/4 Pliki audio już istnieją, pomijam pobieranie i rozpakowywanie.")
    # 2. Transkrypcja z cache
    transcripts = []
    for audio in audio_files:
        print(f"2/4 Przetwarzanie: {audio.name}")
        text = get_transcript(audio)
        if "arkadiusz" in text.lower():
            continue
        transcripts.append(text)
    combined = "\n".join(transcripts)
    # 3. Ekstrakcja fragmentów
    subject = None
    location = None
    for line in combined.split("\n"):
        lw = line.lower()
        if subject is None and ("informatyka" in lw or "matematyka" in lw):
            subject = line.strip()
        if location is None and "krakowie" in lw:
            location = line.strip()
        if subject and location:
            break
    fragments = []
    if subject:  fragments.append(subject)
    if location: fragments.append(location)
    if not fragments:
        last_two = combined.split("\n")[-2:]
        fragments = [l.strip() for l in last_two if l.strip()]
    fragments = "\n".join(fragments)
    # 4. Wnioskowanie
    print("3/4 Wnioskuję z pomocą LLM...")
    answer = infer_answer(fragments)
    print(f"Odpowiedź:\n{answer}")
    # 5. Wysłanie raportu
    payload = {"task": "mp3", "apikey": CENTRALA_KEY, "answer": answer}
    resp = requests.post(REPORT_URL, json=payload)
    if resp.status_code == 200:
        print("✅ Odpowiedź wysłana, serwer odpowiedział:", resp.json())
    else:
        print(f"❌ Błąd przy wysyłaniu: {resp.status_code}\n{resp.text}")

if __name__ == "__main__":
    main()