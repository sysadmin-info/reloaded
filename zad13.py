#!/usr/bin/env python3
"""
S03E04 - Znajdowanie Barbary Zawadzkiej
Multi-engine: openai, lmstudio, anything, gemini, claude
Wykorzystuje iteracyjne przeszukiwanie API people/places
"""
import argparse
import os
import sys
import requests
import json
import re
import unicodedata
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, Set, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Filtr do maskowania URL w logach
class URLMaskingFilter(logging.Filter):
    def filter(self, record):
        url_pattern = r"https?://[^\s]+"
        if record.msg:
            record.msg = re.sub(url_pattern, "***HIDDEN_URL***", record.msg)
        return True

logger.addFilter(URLMaskingFilter())

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(
    description="Znajdowanie Barbary Zawadzkiej (multi-engine)"
)
parser.add_argument(
    "--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
    help="LLM backend to use"
)
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
    print(f"❌ Nieobsługiwany silnik: {ENGINE}")
    sys.exit(1)

print(f"🔄 ENGINE wykryty: {ENGINE}")

# Sprawdzenie zmiennych środowiskowych
PEOPLE_URL: str = os.getenv("PEOPLE_URL")
PLACES_URL: str = os.getenv("PLACES_URL")
BARBARA_NOTE_URL: str = os.getenv("BARBARA_NOTE_URL")
REPORT_URL: str = os.getenv("REPORT_URL")
CENTRALA_API_KEY: str = os.getenv("CENTRALA_API_KEY")

if not all([REPORT_URL, CENTRALA_API_KEY]):
    print("❌ Brak wymaganych zmiennych: REPORT_URL, CENTRALA_API_KEY")
    sys.exit(1)

# Konfiguracja modelu
if ENGINE == "openai":
    MODEL_NAME: str = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
elif ENGINE == "claude":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
elif ENGINE == "gemini":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
elif ENGINE == "lmstudio":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
elif ENGINE == "anything":
    MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")

print(f"✅ Model: {MODEL_NAME}")

# Sprawdzenie API keys
if ENGINE == "openai" and not os.getenv("OPENAI_API_KEY"):
    print("❌ Brak OPENAI_API_KEY")
    sys.exit(1)
elif ENGINE == "claude" and not (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
    print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY")
    sys.exit(1)
elif ENGINE == "gemini" and not os.getenv("GEMINI_API_KEY"):
    print("❌ Brak GEMINI_API_KEY")
    sys.exit(1)

# 2. Inicjalizacja spaCy
try:
    import spacy
    nlp = spacy.load("pl_core_news_lg")
    USE_SPACY = True
    logger.info("SpaCy załadowane pomyślnie")
except Exception as e:
    logger.warning(f"Nie można załadować spaCy: {e}")
    USE_SPACY = False

# 3. Inicjalizacja klienta LLM
def call_llm(prompt: str, temperature: float = 0) -> str:
    """Uniwersalna funkcja wywołania LLM"""
    
    if ENGINE == "openai":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_URL') or None
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
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
            max_tokens=1000
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
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    
    elif ENGINE == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            [prompt],
            generation_config={"temperature": temperature, "max_output_tokens": 1000}
        )
        return response.text.strip()

# 4. Typowanie stanu pipeline
class PipelineState(TypedDict, total=False):
    barbara_note: str
    people_to_check: List[str]
    places_to_check: List[str]
    checked_people: Set[str]
    checked_places: Set[str]
    barbara_locations: List[str]  # Lista wszystkich miejsc gdzie była Barbara
    result: Optional[str]

# 5. Funkcje pomocnicze
def normalize_query(query: str) -> str:
    """Normalizuje zapytanie do formy mianownikowej i usuwa polskie znaki"""
    if USE_SPACY:
        try:
            doc = nlp(query)
            normalized_tokens = [token.lemma_.upper() for token in doc if token.is_alpha]
            if normalized_tokens:
                normalized = normalized_tokens[0]
            else:
                normalized = query.upper()
        except:
            normalized = query.upper()
    else:
        # Fallback - tylko uppercase
        normalized = query.upper()
    
    # Usunięcie polskich znaków diakrytycznych
    normalized_ascii = (
        unicodedata.normalize("NFKD", normalized)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    
    # Usunięcie znaków innych niż litery A-Z
    normalized_ascii = ''.join([char for char in normalized_ascii if char.isalpha()])
    
    logger.debug(f"Znormalizowane słowo: '{query}' -> '{normalized_ascii}'")
    return normalized_ascii

def query_api(url: str, query: str) -> Optional[Dict[str, Any]]:
    """Wysyła zapytanie do API"""
    payload = {
        "apikey": CENTRALA_API_KEY,
        "query": query
    }
    
    try:
        safe_payload = payload.copy()
        safe_payload['apikey'] = "***HIDDEN***"
        logger.info(f"Payload wysyłany do {url}: {safe_payload}")
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        logger.info(f"Odpowiedź serwera dla '{query}': {response.text}")
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, dict) and result.get("code") == 0:
            return result
        else:
            logger.error(f"API zwróciło błąd: {result}")
            return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Błąd podczas zapytania do API {url} z query '{query}': {e}")
        return {}

def extract_keywords(text: str) -> tuple[Set[str], Set[str]]:
    """Wydobywa potencjalne imiona i nazwy miast z notatki"""
    keywords = set()
    
    if USE_SPACY:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in {'persName', 'placeName', 'person', 'geogName', 'loc', 'PER', 'GPE', 'LOC'}:
                keywords.add(ent.text)
    else:
        # Fallback - wyrażenia regularne
        # Szukamy słów zaczynających się od wielkiej litery
        pattern = r'\b[A-ZŁŚŻŹĆŃ][a-ząęółśżźćń]+(?:\s+[A-ZŁŚŻŹĆŃ][a-ząęółśżźćń]+)?\b'
        matches = re.findall(pattern, text)
        keywords.update(matches)
    
    logger.info(f"Znalezione słowa kluczowe: {keywords}")
    
    # Rozdzielamy na imiona i miasta na podstawie kontekstu
    people = set()
    places = set()
    
    # Znane imiona z zadania
    known_people = {'Barbara', 'Aleksander', 'Rafał', 'Andrzej', 'Maja', 'Aleksandra'}
    known_places = {'Warszawa', 'Kraków', 'Grudziądz', 'Lublin'}
    
    for kw in keywords:
        kw_lower = kw.lower()
        if any(name.lower() in kw_lower for name in known_people):
            people.add(kw)
        elif any(place.lower() in kw_lower for place in known_places):
            places.add(kw)
        # Dodaj słowa które wyglądają jak imiona (pojedyncze słowa z wielkiej litery)
        elif len(kw.split()) == 1 and kw[0].isupper():
            people.add(kw)
    
    # Dodaj explicite wymienione osoby
    if "Rafał Bomba" in text:
        people.add("Rafał")
    if "Barbara Zawadzka" in text or "Barbary Zawadzkiej" in text:
        people.add("Barbara")
    if "Aleksander" in text or "Aleksandra Ragowskiego" in text:
        people.add("Aleksander")
    if "Andrzej" in text or "Andrzejem Majem" in text:
        people.add("Andrzej")
        people.add("Maja")
    
    # Dodaj miasta wspomniane w tekście
    for city in ["Warszawa", "Kraków", "Grudziądz", "Lublin"]:
        if city.lower() in text.lower():
            places.add(city)
    
    return people, places

# 6. Nodes dla LangGraph
def download_note_node(state: PipelineState) -> PipelineState:
    """Pobiera notatkę o Barbarze"""
    logger.info("Rozpoczynam poszukiwania Barbary Zawadzkiej...")
    
    try:
        response = requests.get(BARBARA_NOTE_URL)
        response.raise_for_status()
        state["barbara_note"] = response.text
        logger.info("Notatka Barbary pobrana pomyślnie.")
    except Exception as e:
        logger.error(f"Błąd pobierania notatki: {e}")
        state["barbara_note"] = ""
    
    return state

def extract_keywords_node(state: PipelineState) -> PipelineState:
    """Ekstraktuje imiona i miasta z notatki"""
    note = state.get("barbara_note", "")
    if not note:
        logger.error("Brak notatki do analizy")
        return state
    
    # Ekstraktuj słowa kluczowe
    people, places = extract_keywords(note)
    
    # Normalizuj
    normalized_people = {normalize_query(p) for p in people}
    normalized_places = {normalize_query(p) for p in places}
    
    # Usuń puste i zbyt krótkie
    normalized_people = {p for p in normalized_people if len(p) > 2}
    normalized_places = {p for p in normalized_places if len(p) > 2}
    
    logger.info(f"Znormalizowane poszlaki: {normalized_people | normalized_places}")
    
    # Konwertuj na listy (kolejki)
    state["people_to_check"] = list(normalized_people)
    state["places_to_check"] = list(normalized_places)
    state["checked_people"] = set()
    state["checked_places"] = set()
    state["barbara_locations"] = []
    
    return state

def search_loop_node(state: PipelineState) -> PipelineState:
    """Główna pętla wyszukiwania"""
    people_queue = state.get("people_to_check", [])
    places_queue = state.get("places_to_check", [])
    checked_people = state.get("checked_people", set())
    checked_places = state.get("checked_places", set())
    barbara_locations = state.get("barbara_locations", [])
    
    # Określ znane lokalizacje Barbary z notatki
    # Na podstawie analizy: Barbara była widziana w Warszawie i Krakowie (z notatki)
    known_barbara_locations = {"WARSZAWA", "KRAKOW"}
    new_barbara_location = None
    
    while people_queue or places_queue:
        # Sprawdź osoby
        if people_queue:
            person = people_queue.pop(0)
            if person not in checked_people:
                logger.debug(f"Sprawdzam: {person}")
                response = query_api(PEOPLE_URL, person)
                checked_people.add(person)
                
                if response and response.get("code") == 0:
                    message = response.get("message", "")
                    places = message.split()
                    for place in places:
                        normalized_place = normalize_query(place)
                        if normalized_place not in checked_places and normalized_place not in places_queue:
                            places_queue.append(normalized_place)
        
        # Sprawdź miejsca
        if places_queue:
            place = places_queue.pop(0)
            if place not in checked_places:
                logger.debug(f"Sprawdzam: {place}")
                response = query_api(PLACES_URL, place)
                checked_places.add(place)
                
                if response and response.get("code") == 0:
                    message = response.get("message", "")
                    people = message.split()
                    
                    # Sprawdź czy Barbara jest w tym miejscu
                    if "BARBARA" in people:
                        logger.info(f"Znaleziono Barbarę w miejscu: {place}")
                        barbara_locations.append(place)
                        
                        # Sprawdź czy to NOWE miejsce (nie z notatki)
                        if place not in known_barbara_locations:
                            new_barbara_location = place
                            logger.info(f"🎯 To jest NOWA lokalizacja Barbary: {place}")
                            # Kontynuuj przeszukiwanie, może być więcej miejsc
                        else:
                            logger.info(f"ℹ️ {place} to znana lokalizacja z notatki")
                    
                    # Dodaj nowe osoby do kolejki
                    for person_name in people:
                        normalized_person = normalize_query(person_name)
                        if normalized_person not in checked_people and normalized_person not in people_queue:
                            people_queue.append(normalized_person)
    
    logger.info("Zakończono przeszukiwanie.")
    
    if new_barbara_location:
        state["result"] = new_barbara_location
        logger.info(f"✅ Barbara znajduje się w nowej lokalizacji: {new_barbara_location}")
    elif barbara_locations:
        # Jeśli nie znaleziono nowej lokalizacji, ale były jakieś lokalizacje
        logger.warning(f"Nie znaleziono NOWEJ lokalizacji. Znane lokalizacje: {barbara_locations}")
        state["result"] = None
    else:
        logger.warning("Nie udało się znaleźć Barbary.")
        state["result"] = None
    
    if barbara_locations:
        logger.info(f"Wszystkie lokalizacje gdzie widziano Barbarę: {barbara_locations}")
    
    return state

def send_answer_node(state: PipelineState) -> PipelineState:
    """Wysyła odpowiedź do centrali"""
    location = state.get("result")
    if not location:
        logger.error("Brak lokalizacji do wysłania")
        return state
    
    payload = {
        "task": "loop",
        "apikey": CENTRALA_API_KEY,
        "answer": location
    }
    
    logger.info(f"Wysyłam odpowiedź: {location}")
    
    try:
        response = requests.post(REPORT_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Odpowiedź centrali: {result}")
        
        # Sprawdź czy to jest poprawna odpowiedź
        if result.get("code") == 0 and "FLG" in str(result):
            print(f"🏁 {result.get('message', result)}")
        elif result.get("code") != 0:
            logger.warning(f"Centrala odrzuciła odpowiedź: {result}")
            # Może spróbować z inną lokalizacją?
            barbara_locations = state.get("barbara_locations", [])
            if len(barbara_locations) > 1:
                logger.info(f"Inne możliwe lokalizacje: {barbara_locations}")
                
    except Exception as e:
        logger.error(f"Błąd wysyłania: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Szczegóły: {e.response.text}")
    
    return state

def build_graph() -> Any:
    """Buduje graf LangGraph"""
    graph = StateGraph(state_schema=PipelineState)
    
    # Dodaj nodes
    graph.add_node("download_note", download_note_node)
    graph.add_node("extract_keywords", extract_keywords_node)
    graph.add_node("search_loop", search_loop_node)
    graph.add_node("send_answer", send_answer_node)
    
    # Dodaj edges
    graph.add_edge(START, "download_note")
    graph.add_edge("download_note", "extract_keywords") 
    graph.add_edge("extract_keywords", "search_loop")
    graph.add_edge("search_loop", "send_answer")
    graph.add_edge("send_answer", END)
    
    return graph.compile()

def main() -> None:
    print("=== Zadanie 15: Znajdowanie Barbary Zawadzkiej ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔧 Model: {MODEL_NAME}")
    print(f"📚 SpaCy: {'TAK' if USE_SPACY else 'NIE (fallback na regex)'}")
    print("Startuje pipeline...\n")
    
    try:
        graph = build_graph()
        result: PipelineState = graph.invoke({})
        
        if result.get("result"):
            print(f"\n🎉 Zadanie zakończone! Barbara znajduje się w: {result['result']}")
        else:
            print("\n❌ Nie udało się znaleźć Barbary")
            
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
