#!/usr/bin/env python3
"""
S05E02 - Odtworzenie agenta GPS
Multi-engine: openai, lmstudio, anything, gemini, claude
Wykorzystuje API do namierzania osób na podstawie sygnału GPS
"""
import argparse
import os
import sys
import json
import requests
import re
import unicodedata
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, Set, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Agent GPS - namierzanie osób (multi-engine)")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
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
GPS_URL: str = os.getenv("GPS_URL")
GPS_QUESTIONS: str = os.getenv("GPS_QUESTIONS")
APIDB_URL: str = os.getenv("APIDB_URL")
PLACES_URL: str = os.getenv("PLACES_URL")
REPORT_URL: str = os.getenv("REPORT_URL")
CENTRALA_API_KEY: str = os.getenv("CENTRALA_API_KEY")

if not all([REPORT_URL, CENTRALA_API_KEY, APIDB_URL, PLACES_URL]):
    print("❌ Brak wymaganych zmiennych: REPORT_URL, CENTRALA_API_KEY, APIDB_URL, PLACES_URL", file=sys.stderr)
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
    print("❌ Brak OPENAI_API_KEY", file=sys.stderr)
    sys.exit(1)
elif ENGINE == "claude" and not (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
    print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY", file=sys.stderr)
    sys.exit(1)
elif ENGINE == "gemini" and not os.getenv("GEMINI_API_KEY"):
    print("❌ Brak GEMINI_API_KEY", file=sys.stderr)
    sys.exit(1)

# 2. Inicjalizacja klienta LLM
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

# 3. Typowanie stanu pipeline
class PipelineState(TypedDict, total=False):
    gps_logs: str
    gps_question: Dict[str, Any]
    target_city: str
    people_in_city: List[str]
    people_ids: Dict[str, int]  # {name: user_id}
    gps_coordinates: Dict[str, Dict[str, float]]  # {name: {lat, lon}}
    flag: Optional[str]

# 4. Funkcje pomocnicze
def normalize_query(query: str) -> str:
    """Normalizuje zapytanie do formy mianownikowej i usuwa polskie znaki"""
    # Uppercase
    normalized = query.upper()
    
    # Usunięcie polskich znaków diakrytycznych
    normalized_ascii = (
        unicodedata.normalize("NFKD", normalized)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    
    # Usunięcie znaków innych niż litery A-Z
    normalized_ascii = ''.join([char for char in normalized_ascii if char.isalpha()])
    
    return normalized_ascii

def make_db_request(query: str) -> Optional[List[Dict[str, Any]]]:
    """Wykonuje zapytanie do API bazy danych"""
    payload = {
        "task": "database",
        "apikey": CENTRALA_API_KEY,
        "query": query
    }
    
    print(f"📤 Wysyłam zapytanie SQL: {query}")
    
    try:
        response = requests.post(APIDB_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "reply" in result and result["reply"] is not None:
            print(f"✅ Otrzymano odpowiedź z bazy danych")
            return result["reply"]
        else:
            print(f"⚠️  API zwróciło nieoczekiwaną odpowiedź: {result}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd podczas wykonywania zapytania: {e}")
        return None

def query_places_api(city: str) -> Optional[List[str]]:
    """Wysyła zapytanie do API places"""
    payload = {
        "apikey": CENTRALA_API_KEY,
        "query": city
    }
    
    try:
        print(f"📤 Sprawdzam osoby w mieście: {city}")
        response = requests.post(PLACES_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if result.get("code") == 0 and "message" in result:
            people = result["message"].split()
            print(f"✅ Znaleziono {len(people)} osób w {city}")
            return people
        else:
            print(f"⚠️  API places zwróciło błąd: {result}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd podczas zapytania do API places: {e}")
        return []

def get_gps_coordinates(user_id: int) -> Optional[Dict[str, float]]:
    """Pobiera koordynaty GPS dla użytkownika"""
    # Endpoint GPS to /gps w centrali
    gps_url = REPORT_URL.replace('/report', '/gps')
    
    payload = {
        "userID": user_id
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"📍 Pobieram GPS dla user_id: {user_id}")
        response = requests.post(gps_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # Sprawdź różne możliwe formaty odpowiedzi
        if "message" in result:
            msg = result["message"]
            if isinstance(msg, dict) and "lat" in msg and "lon" in msg:
                coords = {"lat": msg["lat"], "lon": msg["lon"]}
                print(f"✅ GPS: lat={coords['lat']}, lon={coords['lon']}")
                return coords
        
        print(f"⚠️  Nieoczekiwany format GPS: {result}")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd podczas pobierania GPS: {e}")
        return None

# 5. Nodes dla LangGraph
def download_data_node(state: PipelineState) -> PipelineState:
    """Pobiera logi GPS i pytanie od centrali"""
    print("\n📥 Pobieram dane wejściowe...")
    
    # Pobierz logi (opcjonalne - do analizy)
    if GPS_URL:
        try:
            response = requests.get(GPS_URL)
            response.raise_for_status()
            state["gps_logs"] = response.text
            print("✅ Pobrano logi GPS")
        except Exception as e:
            print(f"⚠️  Nie udało się pobrać logów: {e}")
            state["gps_logs"] = ""
    
    # Pobierz pytanie
    try:
        response = requests.get(GPS_QUESTIONS)
        response.raise_for_status()
        question_data = response.json()
        state["gps_question"] = question_data
        print(f"✅ Pobrano pytanie: {question_data}")
    except Exception as e:
        print(f"❌ Błąd pobierania pytania: {e}")
        state["gps_question"] = {}
    
    return state

def analyze_question_node(state: PipelineState) -> PipelineState:
    """Analizuje pytanie i wydobywa miasto"""
    print("\n🔍 Analizuję pytanie...")
    
    question = state.get("gps_question", {})
    
    # Sprawdź różne możliwe struktury pytania
    if "question" in question:
        # Może być pytanie o miasto
        q_text = question["question"]
        print(f"Pytanie: {q_text}")
        
        # Użyj LLM do wydobycia miasta z pytania
        prompt = f"""Z poniższego pytania wydobądź nazwę miasta. Zwróć TYLKO nazwę miasta, nic więcej:

Pytanie: {q_text}

Nazwa miasta:"""
        
        city = call_llm(prompt).strip()
        state["target_city"] = normalize_query(city)
        print(f"✅ Wykryte miasto: {state['target_city']}")
        
    elif isinstance(question, str):
        # Może być bezpośrednio nazwa miasta
        state["target_city"] = normalize_query(question)
        print(f"✅ Miasto z pytania: {state['target_city']}")
    else:
        # Domyślnie spróbuj różne pola
        for field in ["city", "location", "place", "target"]:
            if field in question:
                state["target_city"] = normalize_query(question[field])
                print(f"✅ Miasto z pola '{field}': {state['target_city']}")
                break
    
    if not state.get("target_city"):
        print("❌ Nie udało się wykryć miasta z dostępnych danych")
        raise ValueError("Brak miasta do przetworzenia")
    
    return state

def get_people_in_city_node(state: PipelineState) -> PipelineState:
    """Pobiera listę osób w danym mieście"""
    print("\n👥 Pobieram osoby w mieście...")
    
    city = state.get("target_city", "")
    if not city:
        print("❌ Brak miasta do sprawdzenia")
        state["people_in_city"] = []
        return state
    
    people = query_places_api(city)
    if people:
        # Normalizuj imiona i odfiltruj Barbarę
        normalized_people = []
        for person in people:
            normalized = normalize_query(person)
            if normalized != "BARBARA":  # Pomijamy Barbarę zgodnie z instrukcją
                normalized_people.append(normalized)
            else:
                print("ℹ️  Pomijam Barbarę zgodnie z instrukcją")
        
        state["people_in_city"] = normalized_people
        print(f"✅ Osoby do namierzenia: {normalized_people}")
    else:
        state["people_in_city"] = []
    
    return state

def get_user_ids_node(state: PipelineState) -> PipelineState:
    """Pobiera ID użytkowników z bazy danych"""
    print("\n🔢 Pobieram ID użytkowników z bazy...")
    
    people = state.get("people_in_city", [])
    people_ids = {}
    
    for person in people:
        # Zapytanie SQL do pobrania ID użytkownika
        sql_query = f"SELECT id, username FROM users WHERE username = '{person}'"
        result = make_db_request(sql_query)
        
        if result and len(result) > 0:
            user_id = result[0].get("id")
            if user_id:
                people_ids[person] = user_id
                print(f"✅ {person}: ID = {user_id}")
        else:
            print(f"⚠️  Nie znaleziono ID dla: {person}")
    
    state["people_ids"] = people_ids
    return state

def get_gps_data_node(state: PipelineState) -> PipelineState:
    """Pobiera koordynaty GPS dla każdej osoby"""
    print("\n📡 Pobieram dane GPS...")
    
    people_ids = state.get("people_ids", {})
    gps_coordinates = {}
    
    for person, user_id in people_ids.items():
        coords = get_gps_coordinates(user_id)
        if coords:
            gps_coordinates[person] = coords
            print(f"✅ GPS dla {person}: {coords}")
        else:
            print(f"⚠️  Brak GPS dla {person}")
    
    state["gps_coordinates"] = gps_coordinates
    return state

def send_answer_node(state: PipelineState) -> PipelineState:
    """Wysyła odpowiedź do centrali"""
    print("\n📤 Wysyłam odpowiedź do centrali...")
    
    gps_data = state.get("gps_coordinates", {})
    
    if not gps_data:
        print("❌ Brak danych GPS do wysłania")
        return state
    
    # Przygotuj odpowiedź w wymaganym formacie
    # Używamy oryginalnych imion (małe litery)
    answer = {}
    for person, coords in gps_data.items():
        # Przekonwertuj nazwę z powrotem na małe litery
        person_lower = person.lower()
        answer[person_lower] = coords
    
    payload = {
        "task": "gps",
        "apikey": CENTRALA_API_KEY,
        "answer": answer
    }
    
    print(f"📤 Wysyłam dane GPS dla {len(answer)} osób")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(REPORT_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Odpowiedź centrali: {result}")
        
        # Sprawdź czy jest flaga
        if "message" in result:
            message = result["message"]
            if isinstance(message, str) and "FLG" in message:
                state["flag"] = message
                print(f"🏁 FLAGA: {message}")
        
    except Exception as e:
        print(f"❌ Błąd wysyłania: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Szczegóły: {e.response.text}")
    
    return state

def build_graph() -> Any:
    """Buduje graf LangGraph"""
    graph = StateGraph(state_schema=PipelineState)
    
    # Dodaj nodes
    graph.add_node("download_data", download_data_node)
    graph.add_node("analyze_question", analyze_question_node)
    graph.add_node("get_people_in_city", get_people_in_city_node)
    graph.add_node("get_user_ids", get_user_ids_node)
    graph.add_node("get_gps_data", get_gps_data_node)
    graph.add_node("send_answer", send_answer_node)
    
    # Dodaj edges
    graph.add_edge(START, "download_data")
    graph.add_edge("download_data", "analyze_question")
    graph.add_edge("analyze_question", "get_people_in_city")
    graph.add_edge("get_people_in_city", "get_user_ids")
    graph.add_edge("get_user_ids", "get_gps_data")
    graph.add_edge("get_gps_data", "send_answer")
    graph.add_edge("send_answer", END)
    
    return graph.compile()

def main() -> None:
    print("=== Zadanie S05E01: Agent GPS ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔧 Model: {MODEL_NAME}")
    print("Startuje pipeline...\n")
    
    try:
        graph = build_graph()
        result: PipelineState = graph.invoke({})
        
        if result.get("flag"):
            print(f"\n🎉 Zadanie zakończone! Flaga: {result['flag']}")
        else:
            print("\n✅ Zadanie wykonane, czekam na flagę...")
            
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()