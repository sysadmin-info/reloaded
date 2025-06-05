#!/usr/bin/env python3
"""
S03E03 - Zapytania do bazy danych BanAN
Multi-engine: openai, lmstudio, anything, gemini, claude
Wykorzystuje API do wykonywania zapytań SQL i wyszukiwania datacenter z nieaktywnymi menadżerami
"""
import argparse
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, Any, List, Dict
from langgraph.graph import StateGraph, START, END

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Analiza bazy danych BanAN (multi-engine)")
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
APIDB_URL: str = os.getenv("APIDB_URL")
REPORT_URL: str = os.getenv("REPORT_URL")
CENTRALA_API_KEY: str = os.getenv("CENTRALA_API_KEY")

if not all([REPORT_URL, CENTRALA_API_KEY]):
    print("❌ Brak wymaganych zmiennych: REPORT_URL, CENTRALA_API_KEY", file=sys.stderr)
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
    tables: List[str]
    table_schemas: Dict[str, str]
    sql_query: str
    query_result: List[Dict[str, Any]]
    datacenter_ids: List[int]

# 4. Funkcje pomocnicze
def make_db_request(query: str) -> Optional[Dict[str, Any]]:
    """Wykonuje zapytanie do API bazy danych"""
    payload = {
        "task": "database",
        "apikey": CENTRALA_API_KEY,
        "query": query
    }
    
    print(f"📤 Wysyłam zapytanie: {query}")
    
    try:
        response = requests.post(APIDB_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "reply" in result and result["reply"] is not None:
            print(f"✅ Otrzymano odpowiedź")
            return result["reply"]
        else:
            print(f"⚠️  API zwróciło nieoczekiwaną odpowiedź: {result}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd podczas wykonywania zapytania: {e}")
        return None

def extract_sql_from_llm_response(response: str) -> str:
    """Ekstraktuje zapytanie SQL z odpowiedzi LLM"""
    # Usuń ewentualne markdown
    response = response.strip()
    
    # Jeśli jest w bloku kodu
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    
    # Szukaj SELECT
    if "SELECT" in response.upper():
        start = response.upper().find("SELECT")
        # Znajdź koniec zapytania (średnik lub koniec tekstu)
        end = response.find(";", start)
        if end != -1:
            return response[start:end+1].strip()
        else:
            return response[start:].strip()
    
    # Jeśli nic nie znaleziono, zwróć całość
    return response.strip()

# 5. Nodes dla LangGraph
def get_tables_node(state: PipelineState) -> PipelineState:
    """Pobiera listę tabel"""
    print("\n🔍 Pobieram listę tabel...")
    
    result = make_db_request("SHOW TABLES")
    if result:
        tables = [item["Tables_in_banan"] for item in result]
        state["tables"] = tables
        print(f"✅ Znaleziono tabele: {', '.join(tables)}")
    else:
        print("❌ Nie udało się pobrać listy tabel")
        state["tables"] = []
    
    return state

def get_schemas_node(state: PipelineState) -> PipelineState:
    """Pobiera schematy tabel"""
    print("\n📋 Pobieram schematy tabel...")
    
    schemas = {}
    for table in state.get("tables", []):
        print(f"   Pobieram schemat dla: {table}")
        result = make_db_request(f"SHOW CREATE TABLE {table}")
        if result and len(result) > 0:
            schemas[table] = result[0].get("Create Table", "")
            print(f"   ✅ Pobrano schemat dla {table}")
        else:
            print(f"   ⚠️  Nie udało się pobrać schematu dla {table}")
    
    state["table_schemas"] = schemas
    return state

def generate_sql_node(state: PipelineState) -> PipelineState:
    """Generuje zapytanie SQL używając LLM"""
    print("\n🤖 Generuję zapytanie SQL...")
    
    # Przygotuj schematy dla LLM
    schemas_text = ""
    for table, schema in state.get("table_schemas", {}).items():
        schemas_text += f"\nTabela {table}:\n{schema}\n"
    
    prompt = f"""Jesteś ekspertem SQL. Na podstawie poniższych schematów tabel:

{schemas_text}

Napisz zapytanie SQL, które zwróci TYLKO kolumnę DC_ID (identyfikatory) aktywnych datacenter (is_active = 1), 
których menadżerowie są nieaktywni (is_active = 0).

Wskazówki:
- Tabela 'datacenters' zawiera informacje o datacenter (DC_ID, is_active, manager)
- Tabela 'users' zawiera informacje o użytkownikach/menadżerach (id, is_active)
- Kolumna 'manager' w tabeli 'datacenters' odnosi się do 'id' w tabeli 'users'

Zwróć TYLKO surowe zapytanie SQL, bez żadnych wyjaśnień, komentarzy czy formatowania markdown.
Zapytanie musi zwracać TYLKO kolumnę DC_ID."""

    llm_response = call_llm(prompt)
    sql_query = extract_sql_from_llm_response(llm_response)
    
    print(f"📝 Wygenerowane zapytanie SQL:\n{sql_query}")
    state["sql_query"] = sql_query
    
    return state

def execute_query_node(state: PipelineState) -> PipelineState:
    """Wykonuje wygenerowane zapytanie SQL"""
    print("\n⚡ Wykonuję zapytanie SQL...")
    
    sql_query = state.get("sql_query", "")
    if not sql_query:
        print("❌ Brak zapytania SQL do wykonania")
        state["query_result"] = []
        return state
    
    result = make_db_request(sql_query)
    if result:
        state["query_result"] = result
        print(f"✅ Otrzymano {len(result)} wyników")
    else:
        print("❌ Nie udało się wykonać zapytania")
        state["query_result"] = []
    
    return state

def extract_ids_node(state: PipelineState) -> PipelineState:
    """Ekstraktuje ID datacenter z wyników zapytania"""
    print("\n🔢 Ekstraktuję ID datacenter...")
    
    query_result = state.get("query_result", [])
    datacenter_ids = []
    
    for row in query_result:
        # Szukaj klucza zawierającego DC_ID
        for key, value in row.items():
            if "DC_ID" in key.upper() or "dc_id" in key:
                datacenter_ids.append(int(value))
                break
    
    state["datacenter_ids"] = datacenter_ids
    print(f"✅ Znaleziono {len(datacenter_ids)} datacenter: {datacenter_ids}")
    
    return state

def send_answer_node(state: PipelineState) -> PipelineState:
    """Wysyła odpowiedź do centrali"""
    print("\n📡 Wysyłam odpowiedź do centrali...")
    
    datacenter_ids = state.get("datacenter_ids", [])
    
    if not datacenter_ids:
        print("❌ Brak ID datacenter do wysłania")
        return state
    
    payload = {
        "task": "database",
        "apikey": CENTRALA_API_KEY,
        "answer": datacenter_ids
    }
    
    print(f"📤 Wysyłam: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(REPORT_URL, json=payload)
        response.raise_for_status()
        print(f"✅ Odpowiedź centrali: {response.text}")
    except Exception as e:
        print(f"❌ Błąd wysyłania: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Szczegóły: {e.response.text}")
    
    return state

def build_graph() -> Any:
    """Buduje graf LangGraph"""
    graph = StateGraph(state_schema=PipelineState)
    
    # Dodaj nodes
    graph.add_node("get_tables", get_tables_node)
    graph.add_node("get_schemas", get_schemas_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("execute_query", execute_query_node)
    graph.add_node("extract_ids", extract_ids_node)
    graph.add_node("send_answer", send_answer_node)
    
    # Dodaj edges
    graph.add_edge(START, "get_tables")
    graph.add_edge("get_tables", "get_schemas")
    graph.add_edge("get_schemas", "generate_sql")
    graph.add_edge("generate_sql", "execute_query")
    graph.add_edge("execute_query", "extract_ids")
    graph.add_edge("extract_ids", "send_answer")
    graph.add_edge("send_answer", END)
    
    return graph.compile()

def main() -> None:
    print("=== Zadanie 13: Analiza bazy danych BanAN ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔧 Model: {MODEL_NAME}")
    print(f"🌐 API URL: {APIDB_URL}")
    print("Startuje pipeline...\n")
    
    try:
        graph = build_graph()
        result: PipelineState = graph.invoke({})
        
        if result.get("datacenter_ids"):
            print(f"\n🎉 Zadanie zakończone! Znalezione ID: {result['datacenter_ids']}")
        else:
            print("\n❌ Nie udało się znaleźć ID datacenter")
            
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()