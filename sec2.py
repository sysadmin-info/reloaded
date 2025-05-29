#!/usr/bin/env python3
"""
zad14.py - Zapytania do bazy BanAN: tryb wyszukiwania flagi
Multi-engine: openai, lmstudio, anything, gemini, claude
Tryby:
  - flag (domyślnie): wyszukuje i rekonstruuje flagę z tabeli 'correct_order'
  - datacenters: analizuje stan datacenter (z poprzedniego zadania)
"""
import argparse
import os
import sys
import logging
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# -- Konfiguracja -----------------------------------
load_dotenv(override=True)

parser = argparse.ArgumentParser(
    description="Wyszukiwanie flagi"
)
parser.add_argument(
    "--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
    help="LLM backend to use"
)
parser.add_argument(
    "--mode", choices=["flag", "datacenters"], default="flag",
    help="Tryb działania: 'flag' (domyślnie) lub 'datacenters'"
)
args = parser.parse_args()

ENGINE: str = (args.engine or os.getenv("LLM_ENGINE", "openai")).lower()
if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}")
    sys.exit(1)

API_URL: str = os.getenv("APIDB_URL")
API_KEY: str = os.getenv("CENTRALA_API_KEY")
if not API_KEY:
    print("❌ Brak wymaganej zmiennej: CENTRALA_API_KEY")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -- Wysyłanie zapytań do API ----------------------
def make_request(query: str) -> Optional[Any]:
    payload = {"task": "database", "apikey": API_KEY, "query": query}
    logger.debug(f"📤 Wysyłam zapytanie: {query}")
    try:
        resp = requests.post(API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("reply")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Błąd zapytania: {e}")
    return None

# -- Funkcje dla trybu 'flag' -----------------------
def retrieve_tables() -> List[str]:
    result = make_request("SHOW TABLES") or []
    return [item.get("Tables_in_banan") for item in result]

def fetch_rows(table: str) -> List[Dict[str, Any]]:
    return make_request(f"SELECT * FROM {table}") or []

def reconstruct_flag(rows: List[Dict[str, Any]]) -> Optional[str]:
    try:
        sorted_rows = sorted(rows, key=lambda x: int(x.get("weight", 0)))
        msg = "".join(item.get("letter", "") for item in sorted_rows)
        if "FLG:" in msg:
            start = msg.find("FLG:")
            end_idx = msg.find("}", start)
            # exclusive end index: include '}'
            end = end_idx + 1 if end_idx != -1 else len(msg)
            return msg[start:end]
    except Exception as e:
        logger.error(f"❌ Błąd rekonstrukcji flagi: {e}")
    return None


def find_flag():
    logger.info("🔍 Wyszukiwanie flagi...")
    tables = retrieve_tables()
    if "correct_order" not in tables:
        logger.error("⚠️ Brak tabeli 'correct_order'")
        return
    rows = fetch_rows("correct_order")
    if not rows:
        logger.error("⚠️ Brak danych w 'correct_order'")
        return
    raw_flag = reconstruct_flag(rows)
    if not raw_flag:
        logger.warning("⚠️ Nie znaleziono flagi")
        return
    # Usun ewentualna koncowa '}'
    content = raw_flag.rstrip('}')
    # Print w formacie akceptowanym przez agent.py
    print(f"🏁 Flaga znaleziona: {{{{{content}}}}} - kończę zadanie.")

# -- Funkcje dla trybu 'datacenters' ----------------
def analyze_datacenters():
    logger.info("🔧 Analiza nieaktywnych datacenter")
    try:
        from zad13 import build_graph
        graph = build_graph()
        graph.invoke({})
        print("✅ Analiza datacenter zakończona")
    except ImportError:
        logger.error("Nie można zaimportować funkcji build_graph z zad13.py")

# -- Punkt wejścia -------------------------------
if __name__ == "__main__":
    if args.mode == "datacenters":
        analyze_datacenters()
    else:
        find_flag()  
