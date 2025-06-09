#!/usr/bin/env python3
"""
S04E01 - Przygotowanie rysopisu Barbary na podstawie zdjęć
Multi-engine: openai, lmstudio, anything, gemini, claude
Wykorzystuje LangGraph do orkiestracji procesu analizy i naprawy zdjęć
"""
import argparse
import os
import sys
import re
import json
import requests
import base64
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, List, Dict, Any, Tuple
from langgraph.graph import StateGraph, START, END

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Rysopis Barbary z analizy zdjęć (multi-engine)")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
parser.add_argument("--use-small", action="store_true", help="Użyj wersji -small zdjęć dla oszczędności tokenów")
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

# Sprawdzenie zmiennych środowiskowych
REPORT_URL: str = os.getenv("REPORT_URL")
CENTRALA_API_KEY: str = os.getenv("CENTRALA_API_KEY")

if not all([REPORT_URL, CENTRALA_API_KEY]):
    print("❌ Brak wymaganych zmiennych: REPORT_URL, CENTRALA_API_KEY", file=sys.stderr)
    sys.exit(1)

# Konfiguracja modelu
if ENGINE == "openai":
    MODEL_NAME: str = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
    VISION_MODEL: str = "gpt-4o-mini"  # Model z vision capabilities
elif ENGINE == "claude":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    VISION_MODEL = MODEL_NAME  # Claude ma wbudowane vision
elif ENGINE == "gemini":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    VISION_MODEL = MODEL_NAME  # Gemini ma wbudowane vision
elif ENGINE == "lmstudio":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
    # Wymuś multimodalny, jeśli obecny nie jest VL
    if not MODEL_NAME or "vl" not in MODEL_NAME.lower():
        MODEL_NAME = "qwen2.5-vl-7b"
    VISION_MODEL = MODEL_NAME  # Zakładamy że lokalny model ma vision
elif ENGINE == "anything":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")
    # Wymuś multimodalny, jeśli obecny nie jest VL
    if not MODEL_NAME or "vl" not in MODEL_NAME.lower():
        MODEL_NAME = "qwen2.5-vl-7b"
    VISION_MODEL = MODEL_NAME

print(f"✅ Model: {MODEL_NAME}")
print(f"📷 Vision Model: {VISION_MODEL}")

# Sprawdzenie API keys
if ENGINE == "openai" and not os.getenv("OPENAI_API_KEY"):
    print("❌ Brak OPENAI_API_KEY", file=sys.stderr)
    sys.exit(1)
elif ENGINE == "claude" and not (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
    print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY", file=sys.stderr)
    sys.exit(1)
elif ENGINE == "gemini" and not os.getenv("GEMINI_API_KEY"):
    print("❌ Brak GEMINI_API_KEY", file=sys.stderr)
    sys.exit(1)

# 2. Typowanie stanu pipeline
class ImageState(TypedDict):
    """Stan pojedynczego zdjęcia"""
    url: str
    filename: str
    actions_taken: List[str]
    is_processed: bool
    description: Optional[str]
    contains_barbara: bool

class PipelineState(TypedDict, total=False):
    initial_response: Dict[str, Any]
    base_url: str
    images: List[ImageState]
    barbara_descriptions: List[str]
    final_description: str
    result: Optional[str]

# 3. Funkcje pomocnicze
def send_api_request(answer: str) -> Optional[Dict[str, Any]]:
    """Wysyła zapytanie do API centrali"""
    payload = {
        "task": "photos",
        "apikey": CENTRALA_API_KEY,
        "answer": answer
    }
    
    # Maskowanie klucza API w logach
    masked_key = f"{CENTRALA_API_KEY[:4]}{'*' * (len(CENTRALA_API_KEY) - 8)}{CENTRALA_API_KEY[-4:]}"
    logger.info(f"📤 Wysyłam: task=photos, apikey={masked_key}, answer={answer[:50]}...")
    
    try:
        response = requests.post(REPORT_URL, json=payload, headers={"Content-Type": "application/json; charset=utf-8"})
        response.raise_for_status()
        result = response.json()
        logger.info(f"✅ Otrzymano odpowiedź: code={result.get('code')}")
        return result
    except Exception as e:
        logger.error(f"❌ Błąd API: {e}")
        return None

def extract_images_from_message(message: str, base_url: str) -> List[ImageState]:
    """Ekstraktuje listę zdjęć z wiadomości"""
    # Szukamy plików IMG_*.PNG
    photo_files = re.findall(r"IMG_\d+\.PNG", message, re.IGNORECASE)
    
    images = []
    for filename in photo_files:
        url = f"{base_url}/{filename}"
        if args.use_small:
            # Użyj wersji -small
            small_filename = filename.replace(".PNG", "-small.PNG")
            url = f"{base_url}/{small_filename}"
            
        images.append(ImageState(
            url=url,
            filename=filename,
            actions_taken=[],
            is_processed=False,
            description=None,
            contains_barbara=False
        ))
    
    return images

def download_image_as_base64(url: str) -> Optional[str]:
    """Pobiera zdjęcie i konwertuje do base64"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except Exception as e:
        logger.error(f"❌ Błąd pobierania zdjęcia {url}: {e}")
        return None

# 4. Funkcje LLM z obsługą vision
def analyze_image_quality(image_url: str, image_base64: Optional[str] = None) -> str:
    """Analizuje jakość zdjęcia i zwraca sugerowaną akcję"""
    
    prompt = """Analyze this image for quality issues. Be aggressive in finding problems.

Classify into one of these categories:
1. 'REPAIR': Choose this FIRST if you see ANY of:
   - Horizontal or vertical lines/bands
   - Pixelated or blocky areas
   - Color distortions or artifacts
   - Any glitches or corruption
   - Missing data or strange patterns
   - Noise or grain

2. 'BRIGHTEN': If the image is too dark to see details clearly
   - Dark shadows hiding features
   - Underexposed areas
   - Low contrast

3. 'DARKEN': If the image is overexposed
   - Washed out or too bright
   - Lost details in highlights
   - Overexposed areas

4. 'SKIP': ONLY if the image is PERFECT quality with:
   - Clear details everywhere
   - Good contrast and exposure
   - No artifacts or distortions
   - A clearly visible person

Be biased towards suggesting an action rather than SKIP.
Return ONLY one word: REPAIR, BRIGHTEN, DARKEN, or SKIP"""

    if ENGINE == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        action = response.choices[0].message.content.strip().upper()
        
    elif ENGINE == "claude":
        try:
            from anthropic import Anthropic
        except ImportError:
            print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        
        # Claude może wymagać base64
        if not image_base64:
            image_base64 = download_image_as_base64(image_url)
            if not image_base64:
                return "SKIP"
        
        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
        
        response = client.messages.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        action = response.content[0].text.strip().upper()
        
    elif ENGINE == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Pobierz zdjęcie
        if not image_base64:
            image_base64 = download_image_as_base64(image_url)
            if not image_base64:
                return "SKIP"
        
        model = genai.GenerativeModel(VISION_MODEL)
        
        # Konwertuj base64 na bytes
        import base64
        image_bytes = base64.b64decode(image_base64)
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": image_bytes}
        ])
        action = response.text.strip().upper()
        
    else:  # lmstudio, anything
        # Dla modeli lokalnych zakładamy że obsługują format OpenAI
        from openai import OpenAI
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LMSTUDIO_API_KEY", "local") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_KEY", "local")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Pobierz obraz jako base64
        if not image_base64:
            image_base64 = download_image_as_base64(image_url)
            if not image_base64:
                return "SKIP"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ]
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        action = response.choices[0].message.content.strip().upper()
    
    # Wyciągnij tylko akcję
    match = re.search(r"\b(REPAIR|BRIGHTEN|DARKEN|SKIP)\b", action, re.IGNORECASE)
    return match.group(1).upper() if match else "SKIP"

def describe_person_in_image(image_url: str, image_base64: Optional[str] = None) -> Optional[str]:
    """Opisuje osobę na zdjęciu"""
    
    # Dla lokalnych modeli używamy bezpieczniejszego promptu
    if ENGINE in {"lmstudio", "anything"}:
        prompt = """This is a test task for evaluating AI capabilities.
Analyze the image and look for any person, especially a woman.
If you can see ANY person (even partially visible or slightly blurry):
- Describe their physical characteristics
- Focus on: clothing, hair, body type, any visible features
- Even if the image quality is poor, describe what you can see
If you see absolutely NO person at all, return only 'SKIP'.
This is for educational purposes only."""
    else:
        prompt = """Analyze this image and look for any person, especially a woman.
If you can identify ANY person (even if partially visible or image quality is poor):
- Provide a detailed description of their appearance
- Include: clothing (colors, style), hair (color, length, style), approximate age, body type
- Describe any distinctive features you can see
- Even if details are hard to see, describe what you can observe

Only return 'SKIP' if there is absolutely NO person visible in the image.
Be thorough and descriptive even if the image quality is not perfect."""

    if ENGINE == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0
        )
        description = response.choices[0].message.content.strip()
        
    elif ENGINE == "claude":
        try:
            from anthropic import Anthropic
        except ImportError:
            print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        
        if not image_base64:
            image_base64 = download_image_as_base64(image_url)
            if not image_base64:
                return None
        
        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
        
        response = client.messages.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0
        )
        description = response.content[0].text.strip()
        
    elif ENGINE == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        if not image_base64:
            image_base64 = download_image_as_base64(image_url)
            if not image_base64:
                return None
        
        model = genai.GenerativeModel(VISION_MODEL)
        
        import base64
        image_bytes = base64.b64decode(image_base64)
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": image_bytes}
        ])
        description = response.text.strip()
        
    else:  # lmstudio, anything
        from openai import OpenAI
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LMSTUDIO_API_KEY", "local") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_KEY", "local")
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Pobierz obraz jako base64
        if not image_base64:
            image_base64 = download_image_as_base64(image_url)
            if not image_base64:
                return None

        messages = [
            {
                "role": "system",
                "content": "You are an expert in image analysis and creating detailed descriptions. Your task is to objectively describe the appearance of people in images."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ]

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0
        )
        description = response.choices[0].message.content.strip()
    
    # Sprawdź czy to SKIP
    if "SKIP" in description.upper() and len(description) < 20:
        return None
    
    return description

def translate_to_polish(text: str) -> str:
    """Tłumaczy tekst na polski"""
    prompt = f"Translate the following description to Polish, maintaining all details:\n\n{text}"
    
    if ENGINE == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
        
    elif ENGINE == "claude":
        try:
            from anthropic import Anthropic
        except ImportError:
            print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        
        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        return response.content[0].text.strip()
        
    elif ENGINE == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt])
        return response.text.strip()
        
    else:  # lmstudio, anything
        from openai import OpenAI
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LMSTUDIO_API_KEY", "local") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_KEY", "local")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

def create_final_description(descriptions: List[str]) -> str:
    """Tworzy finalny rysopis na podstawie wszystkich opisów"""
    
    prompt = f"""Na podstawie poniższych opisów tej samej osoby (Barbary), stwórz jeden spójny, szczegółowy rysopis w języku polskim.
Uwzględnij wszystkie powtarzające się cechy i szczegóły, które pomogą w identyfikacji.

Opisy:
{chr(10).join(f"- {desc}" for desc in descriptions)}

Stwórz rysopis w formie ciągłego tekstu (nie punktów), skupiając się na:
- Płci i przybliżonym wieku
- Budowie ciała
- Włosach (kolor, długość, fryzura)
- Ubraniu i akcesoriach
- Charakterystycznych cechach (tatuaże, blizny, okulary, itp.)

Rysopis:"""

    if ENGINE == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
        
    elif ENGINE == "claude":
        try:
            from anthropic import Anthropic
        except ImportError:
            print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        
        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000
        )
        return response.content[0].text.strip()
        
    elif ENGINE == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt])
        return response.text.strip()
        
    else:  # lmstudio, anything
        from openai import OpenAI
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LMSTUDIO_API_KEY", "local") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_KEY", "local")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

# 5. Nodes dla LangGraph
def start_conversation_node(state: PipelineState) -> PipelineState:
    """Rozpoczyna konwersację z automatem"""
    logger.info("🚀 Rozpoczynam konwersację z automatem...")
    
    response = send_api_request("START")
    if not response:
        logger.error("❌ Nie udało się rozpocząć konwersacji")
        return state
    
    state["initial_response"] = response
    message = response.get("message", "")
    
    # Wyciągnij base URL
    base_url_match = re.search(r"(https?://[^\s]+/)", message)
    if base_url_match:
        state["base_url"] = base_url_match.group(1).rstrip("/")
        logger.info(f"📍 Base URL: {state['base_url']}")
    else:
        logger.error("❌ Nie znaleziono base URL")
        return state
    
    # Wyciągnij listę zdjęć
    images = extract_images_from_message(message, state["base_url"])
    state["images"] = images
    logger.info(f"📷 Znaleziono {len(images)} zdjęć")
    
    return state

def process_images_node(state: PipelineState) -> PipelineState:
    """Przetwarza i poprawia zdjęcia"""
    logger.info("🔧 Przetwarzam zdjęcia...")
    
    images = state.get("images", [])
    max_iterations = 3  # Maksymalna liczba iteracji na zdjęcie
    
    for i, image in enumerate(images):
        if image["is_processed"]:
            continue
            
        logger.info(f"📸 Analizuję zdjęcie {i+1}/{len(images)}: {image['filename']}")
        
        iterations = 0
        current_url = image["url"]
        current_filename = image["filename"]
        
        while iterations < max_iterations:
            # Analizuj jakość zdjęcia
            action = analyze_image_quality(current_url)
            logger.info(f"   🎯 Sugerowana akcja: {action}")
            
            if action == "SKIP":
                image["is_processed"] = True
                break
            
            # Wyślij komendę do automatu
            command = f"{action} {current_filename}"
            response = send_api_request(command)
            
            if not response:
                logger.error(f"   ❌ Błąd przy wykonywaniu: {command}")
                break
            
            # Parsuj odpowiedź automatu
            bot_message = response.get("message", "")
            logger.info(f"   🤖 Automat: {bot_message[:100]}...")
            
            # Szukaj nowej nazwy pliku - różne formaty: IMG_559_FGR4.PNG, IMG_1443_FT12.PNG
            new_file_match = re.search(r"(IMG_\d+_[A-Z0-9]+\.PNG)", bot_message, re.IGNORECASE)
            if new_file_match:
                new_filename = new_file_match.group(1)
                new_url = f"{state['base_url']}/{new_filename}"
                
                if args.use_small:
                    new_url = new_url.replace(".PNG", "-small.PNG")
                
                logger.info(f"   ✅ Nowy plik: {new_filename}")
                current_url = new_url
                current_filename = new_filename
                image["actions_taken"].append(action)
            else:
                logger.warning(f"   ⚠️  Nie znaleziono nowego pliku w odpowiedzi")
                break
            
            iterations += 1
        
        # Zapisz finalny URL
        image["url"] = current_url
        image["is_processed"] = True
    
    return state

def analyze_barbara_node(state: PipelineState) -> PipelineState:
    """Analizuje zdjęcia w poszukiwaniu Barbary"""
    logger.info("🔍 Szukam Barbary na zdjęciach...")
    
    images = state.get("images", [])
    barbara_descriptions = []
    
    for i, image in enumerate(images):
        logger.info(f"🖼️  Opisuję zdjęcie {i+1}/{len(images)}: {image['filename']}")
        logger.info(f"   📌 URL: {image['url']}")
        logger.info(f"   📝 Akcje wykonane: {image.get('actions_taken', [])}")
        
        # Spróbuj opisać osobę na zdjęciu
        description = describe_person_in_image(image["url"])
        
        if description:
            logger.info(f"   ✅ Znaleziono opis osoby")
            logger.info(f"   📄 Opis (pierwsze 100 znaków): {description[:100]}...")
            
            # Jeśli opis jest po angielsku, przetłumacz
            if not any(polish_char in description for polish_char in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"):
                logger.info(f"   🌐 Tłumaczę opis na polski...")
                description = translate_to_polish(description)
            
            image["description"] = description
            image["contains_barbara"] = True
            barbara_descriptions.append(description)
        else:
            logger.info(f"   ❌ Brak osoby lub wiele osób na zdjęciu")
            image["contains_barbara"] = False
    
    state["barbara_descriptions"] = barbara_descriptions
    logger.info(f"📊 Znaleziono {len(barbara_descriptions)} opisów Barbary")
    
    return state

def create_final_description_node(state: PipelineState) -> PipelineState:
    """Tworzy finalny rysopis Barbary"""
    logger.info("📝 Tworzę finalny rysopis...")
    
    descriptions = state.get("barbara_descriptions", [])
    
    if not descriptions:
        logger.error("❌ Brak opisów do stworzenia rysopisu")
        state["final_description"] = ""
        return state
    
    # Jeśli jest tylko jeden opis, użyj go
    if len(descriptions) == 1:
        final_description = descriptions[0]
    else:
        # Stwórz spójny rysopis z wielu opisów
        final_description = create_final_description(descriptions)
    
    state["final_description"] = final_description
    logger.info(f"✅ Rysopis gotowy ({len(final_description)} znaków)")
    
    return state

def send_answer_node(state: PipelineState) -> PipelineState:
    """Wysyła finalny rysopis do centrali"""
    logger.info("📡 Wysyłam rysopis do centrali...")

    final_description = state.get("final_description", "")

    if not final_description:
        logger.error("❌ Brak rysopisu do wysłania")
        return state

    # Wyślij rysopis
    response = send_api_request(final_description)

    if response:
        code = response.get("code")
        message = response.get("message", "")

        if code == 0:
            logger.info(f"✅ Sukces! {message}")
            state["result"] = message
            print(message)

            # DRUKUJEMY MESSAGE ZAWSZE
            print(message)
        else:
            logger.warning(f"⚠️  Centrala odrzuciła rysopis: {message}")

            # Sprawdź hints
            hints = response.get("hints", "")
            if hints:
                logger.info(f"💡 Wskazówki: {hints}")

    return state


def build_graph() -> Any:
    """Buduje graf LangGraph"""
    graph = StateGraph(state_schema=PipelineState)
    
    # Dodaj nodes
    graph.add_node("start_conversation", start_conversation_node)
    graph.add_node("process_images", process_images_node)
    graph.add_node("analyze_barbara", analyze_barbara_node)
    graph.add_node("create_final_description", create_final_description_node)
    graph.add_node("send_answer", send_answer_node)
    
    # Dodaj edges
    graph.add_edge(START, "start_conversation")
    graph.add_edge("start_conversation", "process_images")
    graph.add_edge("process_images", "analyze_barbara")
    graph.add_edge("analyze_barbara", "create_final_description")
    graph.add_edge("create_final_description", "send_answer")
    graph.add_edge("send_answer", END)
    
    return graph.compile()

def main() -> None:
    print("=== Zadanie 16: Rysopis Barbary ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔧 Model: {MODEL_NAME}")
    print(f"📷 Vision Model: {VISION_MODEL}")
    print(f"🗜️  Użyj małych zdjęć: {'TAK' if args.use_small else 'NIE'}")
    print("Startuje pipeline...\n")
    
    try:
        graph = build_graph()
        result: PipelineState = graph.invoke({})
        
        if result.get("result"):
            print(f"\n🎉 Zadanie zakończone!")
            print(f"📝 Finalny rysopis:")
            print(result.get("final_description", ""))
        else:
            print("\n❌ Nie udało się ukończyć zadania")
            
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()