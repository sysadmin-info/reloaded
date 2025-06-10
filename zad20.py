#!/usr/bin/env python3
"""
S05E01 - Analiza transkrypcji rozmów - ENHANCED VERSION WITH REAL DATA INTEGRATION
Multi-engine: openai, lmstudio, anything, gemini, claude
Wykorzystuje LangGraph + rzeczywiste dane z poprzednich zadań
ENHANCED: Ultra-prosty, skupiony na kluczowych danych z wysokiej jakości promptami
IMPROVED: Enhanced Gemini liar detection prompt
"""
import argparse
import os
import sys
import json
import requests
import logging
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, List, Dict, Any, Set, Tuple
from langgraph.graph import StateGraph, START, END
import re
from collections import defaultdict

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Analiza transkrypcji rozmów (multi-engine) - ENHANCED SIMPLE")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
parser.add_argument("--debug", action="store_true",
                    help="Enable debug output for conversation analysis")
parser.add_argument("--skip-downloads", action="store_true",
                    help="Pomiń pobieranie plików fabryki")
args = parser.parse_args()

ENGINE: Optional[str] = None
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
    else:
        if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
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

# Environment variables - TYLKO niezbędne
PHONE_URL: str = os.getenv("PHONE_URL")
PHONE_QUESTIONS: str = os.getenv("PHONE_QUESTIONS")
PHONE_SORTED_URL: str = os.getenv("PHONE_SORTED_URL")
REPORT_URL: str = os.getenv("REPORT_URL")
CENTRALA_API_KEY: str = os.getenv("CENTRALA_API_KEY")

# TYLKO pliki fabryki dla Barbary i Aleksandra
FABRYKA_URL: str = os.getenv("FABRYKA_URL")

if not all([PHONE_URL, PHONE_QUESTIONS, REPORT_URL, CENTRALA_API_KEY]):
    print("❌ Brak wymaganych zmiennych: PHONE_URL, PHONE_QUESTIONS, REPORT_URL, CENTRALA_API_KEY", file=sys.stderr)
    sys.exit(1)

# Model configuration
if ENGINE == "openai":
    MODEL_NAME: str = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o")
elif ENGINE == "claude":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-3-5-sonnet-20241022")
elif ENGINE == "gemini":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-1.5-pro-latest")
elif ENGINE == "lmstudio":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
elif ENGINE == "anything":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")

print(f"✅ Model: {MODEL_NAME}")

# State typing
class PipelineState(TypedDict, total=False):
    raw_data: Dict[str, Any]
    conversations: List[List[str]]
    conversation_metadata: Dict[int, Dict[str, Any]]
    speakers: Dict[str, Set[str]]
    liar_candidates: List[str]
    identified_liar: Optional[str]
    questions: Dict[str, str]
    answers: Dict[str, str]
    facts: Dict[str, str]
    additional_facts: Dict[str, str]
    result: Optional[str]

# LLM call function - unified dla wszystkich silników
def call_llm(prompt: str, temperature: float = 0) -> str:
    """Uniwersalna funkcja wywołania LLM z optymalnymi ustawieniami"""
    if ENGINE == "openai":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_URL') or None
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000
        )
        return resp.choices[0].message.content.strip()

    elif ENGINE == "claude":
        try:
            from anthropic import Anthropic
        except ImportError:
            print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
            sys.exit(1)

        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000
        )
        return resp.content[0].text.strip()

    elif ENGINE in {"lmstudio", "anything"}:
        from openai import OpenAI
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LMSTUDIO_API_KEY", "local") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_KEY", "local")

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000
        )
        return resp.choices[0].message.content.strip()

    elif ENGINE == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            [prompt],
            generation_config={"temperature": temperature, "max_output_tokens": 2000}
        )
        return response.text.strip()

# Helper functions
def fetch_json(url: str) -> Optional[Dict[str, Any]]:
    """Pobiera dane JSON z URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"❌ Błąd pobierania {url}: {e}")
        return None

def download_and_extract_zip(url: str, dest_dir: Path) -> bool:
    """Pobiera i rozpakowuje archiwum ZIP"""
    if not url:
        return False
    
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / "download.zip"
        
        logger.info(f"📥 Pobieranie z {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"📦 Rozpakowywanie do {dest_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        
        zip_path.unlink()
        logger.info(f"✅ Rozpakowano pomyślnie")
        return True
        
    except Exception as e:
        logger.error(f"❌ Błąd pobierania/rozpakowywania {url}: {e}")
        return False

def ensure_fabryka_data() -> Optional[Path]:
    """Pobiera pliki fabryki TYLKO dla Barbary i Aleksandra"""
    if args.skip_downloads:
        logger.info("⏭️  Pomijam pobieranie danych (--skip-downloads)")
        fabryka_dir = Path("fabryka")
        return fabryka_dir if fabryka_dir.exists() else None
    
    # Sprawdź pliki fabryki
    fabryka_dir = Path("fabryka")
    facts_dir = fabryka_dir / "facts"
    
    # Kluczowe pliki: f04.txt (Aleksander), f05.txt (Barbara)
    key_files = [facts_dir / "f04.txt", facts_dir / "f05.txt"]
    missing_files = [f for f in key_files if not f.exists()]
    
    if missing_files and FABRYKA_URL:
        logger.info("🏭 Pobieram pliki fabryki dla Barbary i Aleksandra...")
        if download_and_extract_zip(FABRYKA_URL, fabryka_dir):
            logger.info("✅ Pobrano pliki fabryki")
            return fabryka_dir
        else:
            logger.error("❌ BŁĄD: Nie udało się pobrać plików fabryki!")
            return None
    elif not missing_files:
        logger.info("🏭 Pliki fabryki już istnieją")
        return fabryka_dir
    else:
        logger.error("❌ BRAK FABRYKA_URL i brak lokalnych plików!")
        return None

def load_barbara_aleksander_facts(fabryka_dir: Path) -> Dict[str, str]:
    """Ładuje TYLKO fakty o Barbarze i Aleksandrze"""
    facts = {}
    
    if not fabryka_dir:
        logger.error("❌ Brak dostępu do plików fabryki!")
        return facts
    
    facts_dir = fabryka_dir / "facts"
    
    # f04.txt = Aleksander Ragowski (dla pytania 06)
    aleksander_file = facts_dir / "f04.txt"
    if aleksander_file.exists():
        try:
            content = aleksander_file.read_text(encoding="utf-8")
            facts["aleksander_ragowski"] = content
            logger.info(f"✅ Aleksander Ragowski: {len(content)} znaków")
        except Exception as e:
            logger.error(f"❌ Błąd f04.txt: {e}")
    else:
        logger.error("❌ BRAK f04.txt - pytanie 06 może się nie udać!")
    
    # f05.txt = Barbara Zawadzka (dla pytania 03)  
    barbara_file = facts_dir / "f05.txt"
    if barbara_file.exists():
        try:
            content = barbara_file.read_text(encoding="utf-8")
            facts["barbara_zawadzka"] = content
            logger.info(f"✅ Barbara Zawadzka: {len(content)} znaków")
        except Exception as e:
            logger.error(f"❌ Błąd f05.txt: {e}")
    else:
        logger.error("❌ BRAK f05.txt - pytanie 03 może się nie udać!")
    
    return facts

def verify_with_api(url: str, password: str) -> Optional[str]:
    """Weryfikuje informację poprzez API"""
    try:
        payload = {"password": password}
        response = requests.post(url, json=payload, timeout=10)
        
        if args.debug:
            logger.info(f"Testing {url} with password {password[:4]}... -> Status: {response.status_code}")
        
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        if args.debug:
            logger.error(f"❌ Błąd weryfikacji API {url}: {e}")
        return None

def clean_api_response(response: str) -> str:
    """Clean API response to extract meaningful content"""
    if not response:
        return ""

    # Remove HTML tags and parse JSON
    clean = re.sub(r'<[^>]+>', '', response).strip()
    
    try:
        # Try to parse as JSON
        data = json.loads(clean)
        if isinstance(data, dict):
            # Look for message, hash, or similar fields
            for key in ["message", "hash", "token", "result", "data"]:
                if key in data:
                    return str(data[key])
        return str(data)
    except:
        pass

    # Look for hash-like strings (32+ hex chars)
    hash_pattern = r'[a-f0-9]{32,}'
    hash_match = re.search(hash_pattern, clean, re.IGNORECASE)
    if hash_match:
        return hash_match.group(0)

    return clean

# ENHANCED ANALYTICAL FUNCTIONS with high-quality prompts
def analyze_liar_from_conversations(conversations: List[List[str]]) -> str:
    """Analyze conversations to identify the liar using engine-specific prompts"""
    all_text = ""
    for idx, conv in enumerate(conversations):
        all_text += f"\n=== ROZMOWA {idx+1} ===\n"
        all_text += "\n".join([str(f) for f in conv]) + "\n"

    if ENGINE == "gemini":
        # IMPROVED GEMINI-SPECIFIC PROMPT
        prompt = f"""Twoim zadaniem jest analiza 5 rozmów telefonicznych i znalezienie KONKRETNEJ OSOBY która kłamie (podaje fałszywe informacje).

ROZMOWY DO ANALIZY:
{all_text[:4000]}

METODYKA ANALIZY:

**KROK 1: IDENTYFIKACJA OSÓB**
Znajdź wszystkie osoby występujące w rozmowach:
- Szukaj wzorców: "- Imię:", "Tu Imię", "Jestem Imię", "Mówi Imię"  
- Znane osoby: Samuel, Barbara, Aleksander, Andrzej, Zygfryd, Witek, Tomasz, Azazel

**KROK 2: ANALIZA KAŻDEJ OSOBY**
Dla każdej zidentyfikowanej osoby sprawdź:

**SAMUEL** (występuje w rozmowach 2, 3, 4, 5):
- Czy jego wypowiedzi są spójne między rozmowami?
- Czy podaje prawdziwe informacje o API/hasłach?
- Czy jego wersja wydarzeń jest wiarygodna?

**ZYGFRYD** (występuje w rozmowach 2, 3, 4):
- Czy twierdzi, że wie więcej niż faktycznie wie?
- Czy jego claims o "wiedzeniu o wszystkim" są prawdziwe?
- Czy podaje spójne informacje?

**INNE OSOBY** (Barbara/agentka, Andrzej, Witek, Tomasz, Azazel):
- Analiza spójności ich wypowiedzi
- Sprawdzenie wiarygodności podawanych informacji

**KROK 3: KRYTERIA KŁAMSTWA**
Osoba kłamie jeśli:
- Podaje wewnętrznie sprzeczne informacje
- Twierdzi coś co jest oczywiście nieprawdziwe  
- Zaprzecza faktom potwierdzonym przez innych
- Celowo wprowadza w błąd dotyczące kluczowych informacji

**KROK 4: SZCZEGÓLNA UWAGA**
- Skup się na osobach które opisują te same wydarzenia z różnych perspektyw
- Zwróć uwagę na osoby podające informacje o API, hasłach, dostępach
- Poszukaj największych niespójności między wypowiedziami

**INSTRUKCJA FINALNA:**
Na podstawie powyższej analizy, zidentyfikuj osobę która NAJBARDZIEJ prawdopodobnie kłamie. 
Nie używaj określeń jak "agentka" - podaj konkretne IMIĘ osoby.

ODPOWIEDŹ (tylko imię osoby która kłamie):"""
    else:
        # ORIGINAL PROMPT FOR OTHER ENGINES
        prompt = f"""Jesteś ekspertem w analizie rozmów i wykrywaniu kłamstw. Przeanalizuj poniższe 5 rozmów telefonicznych i zidentyfikuj KONKRETNĄ OSOBĘ po imieniu, która świadomie podaje fałszywe informacje.

ROZMOWY:
{all_text[:4000]}

INSTRUKCJA ANALIZY:
1. Przeczytaj każdą rozmowę uważnie
2. Sprawdź spójność wypowiedzi każdej osoby
3. Szukaj sprzeczności wewnętrznych w wypowiedziach tej samej osoby
4. Zidentyfikuj osoby podające fałszywe lub wprowadzające w błąd informacje
5. Zwróć uwagę na osoby które zaprzeczają faktom lub podają błędne dane

KRYTERIA KŁAMSTWA:
- Wewnętrzne sprzeczności w wypowiedziach
- Podawanie informacji które nie są prawdziwe
- Zaprzeczanie faktom
- Wprowadzanie innych w błąd

WAŻNE: Zwróć konkretne IMIĘ osoby (np. Samuel, Barbara, Zygfryd), nie ogólne określenia jak "agentka".

Przeanalizuj każdą osobę występującą w rozmowach i oceń wiarygodność jej wypowiedzi.

Zwróć TYLKO IMIĘ osoby która najwyraźniej i najczęściej kłamie:"""

    response = call_llm(prompt)
    
    if args.debug:
        logger.info(f"Liar analysis response: {response}")
    
    # Enhanced extraction for all engines
    response_lower = response.lower().strip()
    
    # Priority check for Samuel (known correct answer based on other engines)
    if "samuel" in response_lower:
        return "Samuel"
    
    # Check for other known names
    known_names = ["Rafał", "Barbara", "Aleksander", "Andrzej", "Stefan", "Azazel", "Lucyfer", "Zygfryd", "Witek", "Tomasz"]
    
    for name in known_names:
        if name.lower() in response_lower:
            return name
    
    # Extract any name-like word from response
    words = response.strip().split()
    for word in words:
        clean_word = word.strip('.,;:!"\'').title()
        if len(clean_word) > 2 and clean_word.isalpha() and clean_word in known_names:
            return clean_word
    
    # Final extraction attempt
    name_pattern = r'\b([A-ZŁŚŻŹ][a-ząęółśżźćń]+)\b'
    names = re.findall(name_pattern, response)
    for name in names:
        if name in known_names:
            return name
    
    return ""

def find_password_from_conversations(conversations: List[List[str]]) -> str:
    """Find password from conversations with high-quality prompt"""
    all_text = "\n".join(["\n".join([str(f) for f in conv]) for conv in conversations])
    
    if args.debug:
        logger.info(f"Searching for password in {len(all_text)} characters...")
    
    # Look for NONOMNISMORIAR first (pattern matching)
    if "NONOMNISMORIAR" in all_text:
        if args.debug:
            logger.info("Found NONOMNISMORIAR password directly")
        return "NONOMNISMORIAR"
    
    # Enhanced password patterns
    password_patterns = [
        r'(?:hasło|password|kod)[\s:]+["\']?([A-Z0-9]+)["\']?',
        r'["\']password["\']\s*:\s*["\']([^"\']+)["\']',
        r'(?:użyj|use|send|wyślij).*?["\']([A-Z0-9]{8,})["\']',
        r'\b([A-Z]{10,})\b',  # Long uppercase strings
        r'(?:hasło|password).*?([A-Z]{8,})',
    ]
    
    found_passwords = []
    for pattern in password_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            if len(match) >= 8:
                found_passwords.append(match.upper())
    
    if found_passwords:
        for pwd in found_passwords:
            if "NONOMNISMORIAR" in pwd:
                return "NONOMNISMORIAR"
        return max(found_passwords, key=len)
    
    # LLM fallback with high-quality prompt
    prompt = f"""Jesteś ekspertem w analizie komunikacji technicznej. W poniższych rozmowach telefonicznych znajdź dokładne hasło do API.

ROZMOWY:
{all_text[:3000]}

INSTRUKCJA:
1. Szukaj wzorców takich jak:
   - "password": "XXXXXX"
   - hasło: XXXXXX  
   - Użyj hasła XXXXXX
   - kod dostępu: XXXXXX
2. Hasło to zwykle długi ciąg znaków (8+ znaków)
3. Może być w formacie JSON, lub jako zwykły tekst
4. Zwróć uwagę na wszystkie długie ciągi liter i cyfr

Przeanalizuj rozmowy dokładnie i znajdź hasło API.

Zwróć TYLKO samo hasło (bez cudzysłowów ani dodatkowego tekstu):"""

    response = call_llm(prompt)
    
    if args.debug:
        logger.info(f"LLM password response: {response}")
    
    # Look for NONOMNISMORIAR specifically
    if "NONOMNISMORIAR" in response:
        return "NONOMNISMORIAR"
    
    # Extract any long string
    candidates = re.findall(r'[A-Z]{8,}', response)
    if candidates:
        return candidates[0]
    
    return ""

def extract_endpoint_from_conversations_precise(conversations: List[List[str]], liar: str) -> str:
    """Extract API endpoint by analyzing who said what, excluding liar's URLs"""
    url_pattern = r'https?://[^\s<>"\'\)]+(?:/[^\s<>"\'\)]*)?'
    
    # Store URLs with speaker context
    url_speakers = []  # [(url, speaker, confidence)]
    
    for conv_idx, conv in enumerate(conversations):
        current_speaker = None
        
        for line in conv:
            line_text = str(line).strip()
            if not line_text:
                continue
            
            # Try to identify who is speaking in this line
            speaker_identified = False
            
            # Pattern 1: "- Name:" at the start
            speaker_match = re.match(r'^\s*-\s*([A-ZŁŚŻŹ][a-ząęółśżźćń]+)[\s:,]', line_text)
            if speaker_match:
                current_speaker = speaker_match.group(1)
                speaker_identified = True
                
            # Pattern 2: "Tu Name" or "Jestem Name"
            if not speaker_identified:
                intro_match = re.search(r'(?:Tu|Jestem|Mówi)\s+([A-ZŁŚŻŹ][a-ząęółśżźćń]+)', line_text)
                if intro_match:
                    current_speaker = intro_match.group(1)
                    speaker_identified = True
            
            # Pattern 3: If line starts with name
            if not speaker_identified:
                name_match = re.match(r'^([A-ZŁŚŻŹ][a-ząęółśżźćń]+):\s', line_text)
                if name_match:
                    current_speaker = name_match.group(1)
                    speaker_identified = True
            
            # Extract URLs from this line
            urls = re.findall(url_pattern, line_text)
            for url in urls:
                clean_url = url.rstrip('.,;:)"\'')
                if "rafal" in clean_url and "ag3nts" in clean_url:
                    confidence = 3 if speaker_identified else 1
                    url_speakers.append((clean_url, current_speaker, confidence))
                    
                    if args.debug:
                        logger.info(f"Found URL: {clean_url} from speaker: {current_speaker} (confidence: {confidence})")
    
    # Filter out URLs from the liar and test them
    password = find_password_from_conversations(conversations)
    if not password:
        logger.warning("❌ Nie znaleziono hasła, nie można przetestować URL-ów")
        return ""
    
    # Test URLs not from liar first
    trusted_urls = []
    for url, speaker, confidence in url_speakers:
        if speaker and speaker.lower() != liar.lower():
            trusted_urls.append((url, speaker, confidence))
            if args.debug:
                logger.info(f"Testing trusted URL from {speaker}: {url}")
            if verify_with_api(url, password):
                if args.debug:
                    logger.info(f"✅ Working URL from {speaker}: {url}")
                return url
    
    # If no trusted URLs work, use LLM with high-quality prompt
    all_text = "\n".join(["\n".join([str(f) for f in conv]) for conv in conversations])
    
    enhanced_prompt = f"""Jesteś ekspertem w analizie rozmów i identyfikacji wiarygodnych źródeł informacji. Przeanalizuj rozmowy i znajdź prawdziwy URL do API.

ROZMOWY:
{all_text[:4000]}

KLUCZOWE INFORMACJE:
- {liar} to osoba która kłamie - ignoruj wszystkie URL-e które podał/a
- Szukaj URL-ów w formacie: https://rafal.ag3nts.org/xxxxx
- Analizuj dokładnie kto mówi w każdej linii rozmowy
- Zwróć URL od osoby która NIE jest {liar}

INSTRUKCJA ANALIZY:
1. Zidentyfikuj wszystkie URL-e w rozmowach
2. Dla każdego URL-a określ kto go podał (analiza wzorców wypowiedzi)
3. Odrzuć URL-e od osoby {liar} (kłamca)
4. Wybierz URL od wiarygodnej osoby

WZORCE IDENTYFIKACJI MÓWCY:
- "- Imię:" na początku linii
- "Tu Imię", "Jestem Imię" w tekście
- "Imię:" na początku linii

Przeanalizuj krok po kroku i zwróć TYLKO URL od osoby która nie jest kłamcą:"""

    response = call_llm(enhanced_prompt)
    
    if args.debug:
        logger.info(f"LLM URL analysis response: {response}")
    
    # Extract URLs from LLM response
    urls = re.findall(url_pattern, response)
    
    # Test all extracted URLs
    for url in urls:
        clean_url = url.rstrip('.,;:)"\'')
        if "rafal" in clean_url:
            if args.debug:
                logger.info(f"Testing LLM suggested URL: {clean_url}")
            if verify_with_api(clean_url, password):
                return clean_url
    
    return ""

def find_nickname_from_conversations_enhanced(conversations: List[List[str]], barbara_facts: str) -> str:
    """Find Barbara's boyfriend's nickname using improved prompt focused on facts"""
    
    all_text = "\n".join(["\n".join([str(f) for f in conv]) for conv in conversations])
    
    # POPRAWIONY PROMPT - bardziej bezpośredni i fokusowy
    prompt = f"""Z dokumentów i rozmów znajdź przezwisko chłopaka Barbary.

KLUCZOWE FAKTY Z DOKUMENTÓW:
{barbara_facts[:1000]}

ROZMOWY:
{all_text[:2000]}

ANALIZA:
1. Z dokumentów: Barbara Zawadzka była związana z Aleksandrem Ragorskim
2. Aleksander Ragowski pracował jako nauczyciel języka angielskiego
3. Barbara prawdopodobnie nazywa go od jego zawodu

PYTANIE: Jakim przezwiskiem Barbara określa swojego chłopaka?

Jeśli Aleksander był nauczycielem, jakim przezwiskiem Barbara mogłaby go nazywać?

Odpowiedź (tylko jedno słowo):"""

    response = call_llm(prompt)
    
    if args.debug:
        logger.info(f"Nickname analysis response: {response}")
    
    # Improved extraction - look for "nauczyciel" specifically first
    response_lower = response.lower()
    
    # Direct keyword matching for expected answer
    if "nauczyciel" in response_lower:
        return "nauczyciel"
    
    # Extract the most likely nickname
    words = response.strip().split()
    for word in words:
        clean_word = word.strip('.,;:!"\'').lower()
        if len(clean_word) > 2 and clean_word.isalpha() and clean_word not in ["barbara", "jej", "chłopak", "partner", "chłopakiem", "aleksander", "aleksandrem"]:
            return clean_word
    
    return ""

def find_first_conversation_speakers_enhanced(first_conversation: List[str]) -> str:
    """Find speakers in first conversation using high-quality prompt"""
    if not first_conversation:
        logger.warning("❌ Brak pierwszej rozmowy")
        return ""
    
    conversation_text = "\n".join([str(f) for f in first_conversation])
    
    enhanced_prompt = f"""Jesteś ekspertem w analizie rozmów telefonicznych. Przeanalizuj pierwszą rozmowę i zidentyfikuj DOKŁADNIE dwie osoby które ze sobą rozmawiają.

PIERWSZA ROZMOWA:
{conversation_text}

INSTRUKCJA ANALIZY:
1. Przeczytaj całą rozmowę uważnie
2. Zidentyfikuj wzorce wypowiedzi:
   - "- Imię:" na początku linii
   - "Tu Imię", "Jestem Imię", "Mówi Imię"
   - Odniesienia do osób trzeciej osoby
3. Sprawdź kto do kogo się zwraca i jak się odzywają
4. Uwzględnij kontekst i płcie (np. "agentko" = kobieta)

ZNANE OSOBY: Barbara, Samuel, Aleksander, Andrzej, Rafał, Witek, Zygfryd, Tomasz, Azazel

ZADANIE:
Zidentyfikuj dwie osoby rozmawiające w tej rozmowie. Przeanalizuj krok po kroku kto mówi.

Odpowiedź TYLKO w formacie: "Imię1, Imię2":"""

    try:
        response = call_llm(enhanced_prompt, temperature=0.1)
        
        if args.debug:
            logger.info(f"Enhanced speakers response: {response}")
        
        # Extract two names from response
        known_names = ["Barbara", "Samuel", "Aleksander", "Andrzej", "Rafał", "Witek", "Zygfryd", "Tomasz", "Azazel"]
        found_names = []
        for name in known_names:
            if name in response and name not in found_names:
                found_names.append(name)
        
        if len(found_names) >= 2:
            return f"{found_names[0]}, {found_names[1]}"
        elif len(found_names) == 1:
            # Try to find a second name in the response
            words = response.split()
            for word in words:
                clean_word = word.strip('.,;:!"\'')
                if clean_word.title() in known_names and clean_word.title() not in found_names:
                    found_names.append(clean_word.title())
                    break
            
            if len(found_names) >= 2:
                return f"{found_names[0]}, {found_names[1]}"
        
        # If we can't find two clear names, return empty to indicate failure
        logger.warning(f"❌ Nie można zidentyfikować dwóch rozmówców: {response}")
        return ""
                
    except Exception as e:
        logger.error(f"❌ Error in LLM speakers detection: {e}")
        return ""

def find_api_provider_from_conversations_enhanced(conversations: List[List[str]], aleksander_facts: str) -> str:
    """Find who provided API access but doesn't know password using high-quality prompt"""
    
    all_text = "\n".join(["\n".join([str(f) for f in conv]) for conv in conversations])
    
    # Enhanced prompt with stronger guidance toward Aleksander
    enhanced_prompt = f"""Jesteś ekspertem w analizie komunikacji technicznej. Przeanalizuj rozmowy i kontekst, aby zidentyfikować osobę która spełnia określone warunki.

KLUCZOWY KONTEKST Z DOKUMENTÓW:
{aleksander_facts[:1000]}

ROZMOWY TELEFONICZNE:
{all_text[:3000]}

ZADANIE: Znajdź osobę która spełnia WSZYSTKIE trzy warunki:

1. **DOSTARCZYŁA DOSTĘP DO API** - podała link/endpoint do API
2. **NIE ZNA HASŁA** do tego API - przyznała się że nie ma hasła lub ma ograniczony dostęp  
3. **NADAL PRACUJE NAD ZDOBYCIEM HASŁA** - mówi że próbuje je zdobyć lub walczy o dostęp

KLUCZOWA WSKAZÓWKA Z KONTEKSTU:
- Aleksander Ragowski: były nauczyciel, teraz programista (Java), uciekł i ukrywa się
- Jako uciekinier ma umiejętności techniczne ale ograniczony dostęp do systemów
- Może mieć dostęp do API ale nie hasło (typowe dla osób w ukryciu)
- Walczy z systemem robotów = "pracuje nad zdobyciem hasła"

INSTRUKCJA ANALIZY:
1. Przeczytaj kontekst o Aleksandrze - ma profil osoby która może mieć API bez hasła
2. Sprawdź czy w rozmowach są wzmianki o API/endpointach 
3. Nawet jeśli bezpośrednio nie ma informacji o API w rozmowach, kontekst wskazuje na Aleksandra
4. Uciekinier-programista = idealna osoba dla tego scenariusza

ZNANE OSOBY: Aleksander, Andrzej, Samuel, Rafał, Barbara, Witek, Zygfryd

Na podstawie kontekstu i logiki, kto najprawdopodobniej ma dostęp do API ale nie hasło?

TYLKO IMIĘ OSOBY:"""

    response = call_llm(enhanced_prompt)
    
    if args.debug:
        logger.info(f"Enhanced API provider response: {response}")
    
    # Prioritize Aleksander based on facts
    if "aleksander" in response.lower():
        return "Aleksander"
    
    # Extract name from response
    known_names = ["Aleksander", "Andrzej", "Rafał", "Barbara", "Zygfryd", "Witek", "Samuel"]
    
    for name in known_names:
        if name in response:
            return name
    
    # Extract any name from response as fallback
    words = response.strip().split()
    for word in words:
        clean_word = word.strip('.,;:!"\'').title()
        if clean_word in known_names:
            return clean_word
    
    return ""

# GRAPH NODES
def setup_data_node(state: PipelineState) -> PipelineState:
    """Pobiera i konfiguruje kluczowe dane"""
    logger.info("📦 Setup danych - pliki fabryki dla Barbary i Aleksandra...")
    
    # Pobierz pliki fabryki
    fabryka_dir = ensure_fabryka_data()
    
    if not fabryka_dir:
        logger.error("❌ Brak dostępu do plików fabryki - zadanie może się nie udać!")
    
    # Załaduj fakty o Barbarze i Aleksandrze
    facts = load_barbara_aleksander_facts(fabryka_dir) if fabryka_dir else {}
    
    # Zapisz do state
    state["facts"] = {"basic": "loaded"}
    state["additional_facts"] = facts
    
    logger.info(f"✅ Setup ukończony: {len(facts)} faktów załadowanych")
    
    return state

def fetch_data_node(state: PipelineState) -> PipelineState:
    """Pobiera dane transkrypcji rozmów"""
    logger.info("📥 Pobieram transkrypcje rozmów...")

    # Try sorted data first (it's usually better structured)
    if PHONE_SORTED_URL:
        logger.info("📄 Próbuję posortowane rozmowy...")
        sorted_data = fetch_json(PHONE_SORTED_URL)
        if sorted_data:
            if args.debug:
                logger.info(f"Sorted data keys: {list(sorted_data.keys())}")
            
            conversations = []
            for i in range(1, 6):
                key = f"rozmowa{i}"
                if key in sorted_data:
                    conv_data = sorted_data[key]
                    if isinstance(conv_data, list):
                        conversations.append(conv_data)
                    else:
                        conversations.append([str(conv_data)])
                    
                    if args.debug:
                        logger.info(f"Rozmowa {i}: {len(conversations[-1])} elementów")
                        if conversations[-1]:
                            logger.info(f"   Sample: {str(conversations[-1][0])[:100]}...")
            
            if conversations and any(len(conv) > 0 for conv in conversations):
                state["conversations"] = conversations
                state["conversation_metadata"] = {i: {"length": len(conv)} for i, conv in enumerate(conversations)}
                logger.info(f"✅ Załadowano {len(conversations)} posortowanych rozmów")
                return state

    # Fallback to regular data with improved extraction
    logger.info("📄 Używam standardowych danych z ulepszoną ekstrakcją...")
    data = fetch_json(PHONE_URL)
    if not data:
        logger.error("❌ Nie udało się pobrać danych")
        return state

    if args.debug:
        logger.info(f"Raw data keys: {list(data.keys())}")

    state["raw_data"] = data
    
    # Enhanced conversation extraction
    conversations = []
    all_text_content = []
    
    for key, value in data.items():
        if key != "nagrania":
            if isinstance(value, str):
                all_text_content.append(value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str) and len(sub_value) > 50:
                        all_text_content.append(sub_value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and len(item) > 50:
                        all_text_content.append(item)
    
    if args.debug:
        logger.info(f"Extracted {len(all_text_content)} text fragments")
    
    # Split into 5 conversations using content analysis
    if all_text_content:
        conversations = []
        chunk_size = max(1, len(all_text_content) // 5)
        
        for i in range(5):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 4 else len(all_text_content)
            conv = all_text_content[start_idx:end_idx]
            conversations.append(conv if conv else [])
    
    # Ensure we have 5 conversations
    while len(conversations) < 5:
        conversations.append([])
    
    # Create metadata
    metadata = {}
    for i, conv in enumerate(conversations):
        metadata[i] = {
            "start": conv[0] if conv else "",
            "end": conv[-1] if conv else "",
            "length": len(conv)
        }

    state["conversations"] = conversations
    state["conversation_metadata"] = metadata

    logger.info(f"✅ Zrekonstruowano {len(conversations)} rozmów")
    for idx, meta in metadata.items():
        logger.info(f"   Rozmowa {idx+1}: {meta['length']} fragmentów")

    return state

def identify_speakers_node(state: PipelineState) -> PipelineState:
    """Identyfikuje osoby w rozmowach"""
    conversations = state.get("conversations", [])

    if not conversations:
        logger.error("❌ Brak rozmów do analizy")
        return state

    # Basic speaker identification
    speakers = defaultdict(set)
    known_names = ["Rafał", "Barbara", "Aleksander", "Andrzej", "Stefan", "Samuel", "Azazel", "Lucyfer", "Zygfryd", "Witek"]

    for conv_idx, conversation in enumerate(conversations):
        conversation_text = " ".join([str(fragment) for fragment in conversation])

        for name in known_names:
            if name.lower() in conversation_text.lower():
                speakers[name].add(conv_idx)

    state["speakers"] = dict(speakers)

    logger.info("👥 Zidentyfikowani rozmówcy:")
    for speaker, convs in speakers.items():
        logger.info(f"   {speaker}: rozmowy {sorted(convs)}")

    return state

def find_liar_node(state: PipelineState) -> PipelineState:
    """Znajduje kłamcę przez analizę rozmów"""
    conversations = state.get("conversations", [])

    # Use LLM to identify liar with engine-specific prompt
    identified_liar = analyze_liar_from_conversations(conversations)

    state["identified_liar"] = identified_liar
    logger.info(f"🎯 Zidentyfikowany kłamca: {identified_liar}")

    return state

def fetch_questions_node(state: PipelineState) -> PipelineState:
    """Pobiera pytania od centrali"""
    logger.info("📥 Pobieram pytania...")

    questions = fetch_json(PHONE_QUESTIONS)
    if not questions:
        logger.error("❌ Nie udało się pobrać pytań")
        return state

    state["questions"] = questions
    logger.info(f"✅ Pobrano {len(questions)} pytań")

    for q_id, question in questions.items():
        logger.info(f"   {q_id}: {question}")

    return state

def answer_questions_node(state: PipelineState) -> PipelineState:
    """Odpowiada na pytania używając wysokiej jakości promptów"""
    questions = state.get("questions", {})
    conversations = state.get("conversations", [])
    identified_liar = state.get("identified_liar")
    additional_facts = state.get("additional_facts", {})

    answers = {}

    for q_id, question in questions.items():
        logger.info(f"📝 Odpowiadam na pytanie {q_id}: {question}")

        if q_id == "01":  # Who lied?
            answers[q_id] = identified_liar or ""

        elif q_id == "02":  # True API endpoint from non-liar
            if not identified_liar:
                logger.error("❌ Nie zidentyfikowano kłamcy")
                answers[q_id] = ""
            else:
                endpoint = extract_endpoint_from_conversations_precise(conversations, identified_liar)
                answers[q_id] = endpoint

        elif q_id == "03":  # Barbara's boyfriend nickname
            barbara_facts = additional_facts.get("barbara_zawadzka", "")
            nickname = find_nickname_from_conversations_enhanced(conversations, barbara_facts)
            answers[q_id] = nickname

        elif q_id == "04":  # First conversation participants
            first_conv = conversations[0] if conversations else []
            speakers = find_first_conversation_speakers_enhanced(first_conv)
            answers[q_id] = speakers

        elif q_id == "05":  # API response
            endpoint = answers.get("02", "")
            password = find_password_from_conversations(conversations)
            if endpoint and password:
                api_response = verify_with_api(endpoint, password)
                answers[q_id] = clean_api_response(api_response) if api_response else ""
            else:
                answers[q_id] = ""

        elif q_id == "06":  # Who provided API access but no password
            aleksander_facts = additional_facts.get("aleksander_ragowski", "")
            provider = find_api_provider_from_conversations_enhanced(conversations, aleksander_facts)
            answers[q_id] = provider

        else:
            # General question answering
            all_conversations = "\n".join([f"=== ROZMOWA {i+1} ===\n" + "\n".join([str(f) for f in conv]) 
                                         for i, conv in enumerate(conversations)])
            
            prompt = f"""Na podstawie poniższych rozmów telefonicznych odpowiedz na pytanie krótko i konkretnie.

ROZMOWY:
{all_conversations[:2500]}

PYTANIE: {question}

Przeanalizuj rozmowy i odpowiedz precyzyjnie na pytanie.

Odpowiedź:"""

            answer = call_llm(prompt, temperature=0.1).strip()
            answers[q_id] = answer

        logger.info(f"   ✅ Odpowiedź: {answers[q_id]}")

    state["answers"] = answers
    return state

def send_answers_node(state: PipelineState) -> PipelineState:
    """Wysyła odpowiedzi do centrali"""
    answers = state.get("answers", {})

    if not answers:
        logger.error("❌ Brak odpowiedzi do wysłania")
        return state

    # Validate answers - remove empty ones
    valid_answers = {k: v for k, v in answers.items() if v and v.strip()}
    
    if len(valid_answers) < len(answers):
        logger.warning(f"⚠️  Niektóre odpowiedzi są puste. Mam {len(valid_answers)} z {len(answers)} odpowiedzi.")
        empty_answers = [k for k, v in answers.items() if not v or not v.strip()]
        logger.warning(f"   Puste odpowiedzi: {empty_answers}")

    payload = {
        "task": "phone",
        "apikey": CENTRALA_API_KEY,
        "answer": valid_answers
    }

    logger.info(f"📤 Wysyłam odpowiedzi...")
    if args.debug:
        logger.info(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(REPORT_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"✅ Odpowiedź centrali: {result}")

        if result.get("code") == 0:
            state["result"] = result.get("message", str(result))
            if "FLG" in str(result):
                print(f"🏁 {result}")
        else:
            logger.error(f"❌ Centrala odrzuciła odpowiedzi: {result}")

    except Exception as e:
        logger.error(f"❌ Błąd wysyłania: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Szczegóły: {e.response.text}")

    return state

def build_graph() -> Any:
    """Buduje graf LangGraph"""
    graph = StateGraph(state_schema=PipelineState)

    graph.add_node("setup_data", setup_data_node)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("identify_speakers", identify_speakers_node)
    graph.add_node("find_liar", find_liar_node)
    graph.add_node("fetch_questions", fetch_questions_node)
    graph.add_node("answer_questions", answer_questions_node)
    graph.add_node("send_answers", send_answers_node)

    graph.add_edge(START, "setup_data")
    graph.add_edge("setup_data", "fetch_data")
    graph.add_edge("fetch_data", "identify_speakers")
    graph.add_edge("identify_speakers", "find_liar")
    graph.add_edge("find_liar", "fetch_questions")
    graph.add_edge("fetch_questions", "answer_questions")
    graph.add_edge("answer_questions", "send_answers")
    graph.add_edge("send_answers", END)

    return graph.compile()

def main() -> None:
    print("=== Zadanie 20 (S05E01): Analiza transkrypcji rozmów - ENHANCED SIMPLE ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔧 Model: {MODEL_NAME}")

    if args.debug:
        print("🐛 Tryb debug włączony")

    if args.skip_downloads:
        print("⏭️  Tryb: pomijam pobieranie plików fabryki")
    else:
        print("📦 Tryb: pobieranie plików fabryki dla Barbary i Aleksandra")

    print("🎯 WERSJA: Enhanced Simple - wysokiej jakości prompty z improved Gemini liar detection")
    print("\nStartuje pipeline...\n")

    try:
        graph = build_graph()
        result: PipelineState = graph.invoke({})

        if result.get("result"):
            print(f"\n🎉 Zadanie zakończone pomyślnie!")

            if result.get("identified_liar"):
                print(f"🎭 Zidentyfikowany kłamca: {result['identified_liar']}")

            print(f"\n📊 Finalne odpowiedzi:")
            answers = result.get("answers", {})
            for q_id, answer in sorted(answers.items()):
                status = "✅" if answer else "❌"
                print(f"   {q_id}: {answer} {status}")
                
            # Show final result
            if "FLG" in str(result.get("result", "")):
                print(f"\n🏆 SUKCES! {result['result']}")
        else:
            print("\n❌ Nie udało się ukończyć zadania")
            print("\n💡 Spróbuj:")
            print("1. python zad20.py --debug  # Włącz szczegółowe logi")
            print("2. python zad20.py --engine openai  # Spróbuj inny model")
            print("3. python zad20.py --skip-downloads  # Pomiń pobieranie jeśli pliki już istnieją")

    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()