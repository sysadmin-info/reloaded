#!/usr/bin/env python3
"""
S03E02 - Wektorowe wyszukiwanie raportów z testów broni
Multi-engine: openai, lmstudio, anything, gemini, claude
Wykorzystuje Qdrant do indeksowania i wyszukiwania semantycznego

Przykładowa konfiguracja .env:
# ...
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import os
import sys
import zipfile
import requests
import json
import re
import uuid
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Any

# Wyłącz ostrzeżenia o embedding
embedding_warning_emitted = False

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Wektorowe wyszukiwanie raportów (multi-engine)")
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
            ENGINE = "lmstudio"  # domyślnie

if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
    sys.exit(1)

print(f"🔄 ENGINE wykryty: {ENGINE}")

# Sprawdzenie zmiennych środowiskowych
FABRYKA_URL: str = os.getenv("FABRYKA_URL")
REPORT_URL: str = os.getenv("REPORT_URL")
CENTRALA_API_KEY: str = os.getenv("CENTRALA_API_KEY")
WEAPONS_PASSWORD: str = os.getenv("WEAPONS_PASSWORD")

if not all([FABRYKA_URL, REPORT_URL, CENTRALA_API_KEY]):
    print("❌ Brak wymaganych zmiennych: FABRYKA_URL, REPORT_URL, CENTRALA_API_KEY", file=sys.stderr)
    sys.exit(1)

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

if ENGINE == "openai" and not os.getenv("OPENAI_API_KEY"):
    print("❌ Brak OPENAI_API_KEY", file=sys.stderr)
    sys.exit(1)
elif ENGINE == "claude" and not (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
    print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY", file=sys.stderr)
    sys.exit(1)
elif ENGINE == "gemini" and not os.getenv("GEMINI_API_KEY"):
    print("❌ Brak GEMINI_API_KEY", file=sys.stderr)
    sys.exit(1)

# 2. Inicjalizacja klienta LLM i embedding
if ENGINE == "openai":
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL_OPENAI", "text-embedding-ada-002")
    VECTOR_SIZE: int = 1536
elif ENGINE == "lmstudio":
    EMBEDDING_MODEL: str = os.getenv("LMSTUDIO_MODEL_EMBED", "text-embedding-nomic-embed-text-v1.5")
    VECTOR_SIZE: int = 768
elif ENGINE == "anything":
    EMBEDDING_MODEL: str = os.getenv("ANYTHING_MODEL_EMBED", "text-embedding-nomic-embed-text-v1.5")
    VECTOR_SIZE: int = 768
else:
    EMBEDDING_MODEL: str = ""
    VECTOR_SIZE: int = 1536

import requests
import os

def get_embedding(text: str) -> Optional[list[float]]:
    global embedding_warning_emitted
    if ENGINE == "openai":
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Błąd generowania embeddingu: {e}")
            return None
    elif ENGINE in {"lmstudio", "anything"}:
        # Składanie URL z końcówką /embeddings
        base_url = os.getenv("LMSTUDIO_API_URL")
        url = base_url.rstrip("/") + "/embeddings"
        model_name = os.getenv("LMSTUDIO_MODEL_EMBED", "nomic-embed-multilingual-v1")
        try:
            headers = {"Content-Type": "application/json"}
            # W LM Studio API_KEY nie jest wymagany, ale zostawiam dla zgodności (niektóre wrappery mogą sprawdzać).
            api_key = os.getenv("LMSTUDIO_API_KEY", "local")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            data = {
                "model": model_name,
                "input": text
            }
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            emb = response.json()
            return emb["data"][0]["embedding"]
        except Exception as e:
            print(f"❌ Błąd embeddingu LM Studio: {e}")
            return None
    else:
        if not embedding_warning_emitted:
            print("⚠️  Wybrany silnik nie obsługuje embeddingów (albo nie zaimplementowano obsługi). Użyj OpenAI lub LM Studio.")
            embedding_warning_emitted = True
        return None


# 3. Inicjalizacja Qdrant
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME: str = "weapons_reports"

if QDRANT_HOST == ":memory:":
    qdrant_client = QdrantClient(":memory:")
    print("📊 Używam Qdrant w pamięci (dane nie będą zapisane)")
elif QDRANT_API_KEY:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=QDRANT_API_KEY
    )
    print(f"☁️  Połączono z Qdrant Cloud")
else:
    qdrant_client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    print(f"🐳 Połączono z Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")

    try:
        qdrant_client.get_collections()
    except Exception as e:
        print(f"❌ Nie mogę połączyć się z Qdrant na {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"   Błąd: {e}")
        print("\n💡 Wskazówki:")
        print("   - Dla Docker: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant")
        print("   - Dla WSL2: upewnij się że Docker Desktop jest uruchomiony")
        print("   - Lub ustaw QDRANT_HOST=:memory: w .env dla trybu in-memory")
        sys.exit(1)

# === USUWANIE KOLEKCJI JEŚLI ISTNIEJE (dowolny tryb, także Cloud/In-Memory) ===
try:
    collections = qdrant_client.get_collections()
    collection_names = [c.name for c in collections.collections]
    if COLLECTION_NAME in collection_names:
        print(f"⚠️  Usuwam istniejącą kolekcję Qdrant '{COLLECTION_NAME}' (konflikt wymiarów embeddingu)...")
        qdrant_client.delete_collection(COLLECTION_NAME)
        print(f"✅ Kolekcja '{COLLECTION_NAME}' została usunięta.")
except Exception as e:
    print(f"❌ Błąd podczas usuwania kolekcji '{COLLECTION_NAME}': {e}")

# 4. Typowanie stanu pipeline
class PipelineState(TypedDict, total=False):
    weapons_dir: Path
    result: str

# 5. Funkcje pomocnicze z typowaniem
def download_and_extract(dest: Path) -> Path:
    """Pobiera i rozpakowuje archiwum z fabryki"""
    dest.mkdir(parents=True, exist_ok=True)

    print(f"📥 Pobieranie plików z {FABRYKA_URL}...")
    main_zip = dest / "fabryka.zip"
    resp = requests.get(FABRYKA_URL, stream=True)
    resp.raise_for_status()

    with open(main_zip, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)

    print("📦 Rozpakowywanie głównego archiwum...")
    with zipfile.ZipFile(main_zip, "r") as zf:
        zf.extractall(dest)

    weapons_zip: Optional[Path] = None
    for file in dest.rglob("weapons_tests.zip"):
        weapons_zip = file
        break

    if not weapons_zip:
        print("❌ Nie znaleziono weapons_tests.zip")
        sys.exit(1)

    print(f"🔓 Rozpakowywanie weapons_tests.zip z hasłem...")
    weapons_dir = weapons_zip.parent / "weapons_tests"

    with zipfile.ZipFile(weapons_zip, "r") as zf:
        zf.extractall(weapons_dir, pwd=WEAPONS_PASSWORD.encode())

    print("✅ Pliki rozpakowane")
    return weapons_dir

def extract_date_from_filename(filename: str) -> Optional[str]:
    """Ekstraktuje datę z nazwy pliku (np. 2024_02_21.txt -> 2024-02-21)"""
    date_match = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", filename)
    if date_match:
        return f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    return None

def create_collection() -> None:
    """Tworzy kolekcję w Qdrant"""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )
    print(f"✅ Utworzono kolekcję '{COLLECTION_NAME}' w Qdrant")

def index_reports(weapons_dir: Path) -> None:
    """Indeksuje raporty w Qdrant"""
    print("🔍 Indeksowanie raportów...")

    points: list[Any] = []
    for file_path in weapons_dir.rglob("*.txt"):
        content: str = file_path.read_text(encoding="utf-8", errors="ignore")
        date: Optional[str] = extract_date_from_filename(file_path.name)
        if not date:
            print(f"⚠️  Nie udało się wyekstraktować daty z: {file_path.name}")
            continue

        embedding: Optional[list[float]] = get_embedding(content)
        if not embedding:
            print(f"⚠️  Nie udało się wygenerować embeddingu dla: {file_path.name}")
            continue

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "date": date,
                "filename": file_path.name,
                "content_preview": content[:200]
            }
        )
        points.append(point)
        print(f"   ✅ Zaindeksowano: {file_path.name} (data: {date})")

    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"✅ Zaindeksowano {len(points)} raportów")
    else:
        print("❌ Brak raportów do zaindeksowania")

def search_theft_report() -> Optional[str]:
    """Wyszukuje raport ze wzmianką o kradzieży prototypu broni"""
    query: str = "W raporcie, z którego dnia znajduje się wzmianka o kradzieży prototypu broni?"

    print(f"🔎 Szukam: {query}")

    query_embedding: Optional[list[float]] = get_embedding(query)
    if not query_embedding:
        print("❌ Nie udało się wygenerować embeddingu dla zapytania")
        return None

    # Używamy .search(), bo .query() z FastEmbed wymaga query_text
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=1,
        with_payload=True
    )

    if results:
        date = results[0].payload.get("date")
        filename = results[0].payload.get("filename")
        preview = results[0].payload.get("content_preview")

        print(f"✅ Znaleziono raport: {filename}")
        print(f"   Data: {date}")
        print(f"   Fragment: {preview}...")

        return date
    else:
        print("❌ Nie znaleziono raportu")
        return None

# 6. Pipeline LangGraph
def download_node(state: PipelineState) -> PipelineState:
    """Node do pobierania i rozpakowywania archiwów"""
    weapons_dir = download_and_extract(Path("fabryka_data"))
    state["weapons_dir"] = weapons_dir
    return state

def index_node(state: PipelineState) -> PipelineState:
    """Node do indeksowania raportów"""
    create_collection()
    index_reports(state["weapons_dir"])
    return state

def search_node(state: PipelineState) -> PipelineState:
    """Node do wyszukiwania raportu"""
    date = search_theft_report()
    state["result"] = date
    return state

def send_node(state: PipelineState) -> PipelineState:
    """Node do wysyłania odpowiedzi"""
    if not state.get("result"):
        print("❌ Brak wyniku do wysłania")
        return state

    payload = {
        "task": "wektory",
        "apikey": CENTRALA_API_KEY,
        "answer": state["result"]
    }

    print(f"\n📤 Wysyłam odpowiedź: {payload['answer']}")

    try:
        resp = requests.post(REPORT_URL, json=payload)
        resp.raise_for_status()
        print(f"✅ Odpowiedź centrali: {resp.text}")
    except Exception as e:
        print(f"❌ Błąd wysyłania: {e}")

    return state

def build_graph() -> Any:
    """Buduje graf LangGraph"""
    graph = StateGraph(state_schema=PipelineState)
    graph.add_node("download", download_node)
    graph.add_node("index", index_node)
    graph.add_node("search", search_node)
    graph.add_node("send", send_node)

    graph.add_edge(START, "download")
    graph.add_edge("download", "index")
    graph.add_edge("index", "search")
    graph.add_edge("search", "send")
    graph.add_edge("send", END)

    return graph.compile()

def main() -> None:
    print("=== Zadanie 12: Wektorowe wyszukiwanie raportów ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔑 Hasło do weapons_tests.zip: {WEAPONS_PASSWORD}")

    if QDRANT_HOST == ":memory:":
        print(f"💾 Qdrant: tryb in-memory (bez persystencji)")
    elif QDRANT_API_KEY:
        print(f"☁️  Qdrant: Cloud")
    else:
        print(f"🐳 Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")

    print("Startuje pipeline...\n")

    try:
        graph = build_graph()
        result: PipelineState = graph.invoke({})

        if result.get("result"):
            print(f"\n🎉 Zadanie zakończone! Odpowiedź: {result['result']}")
        else:
            print("\n❌ Nie udało się znaleźć odpowiedzi")

    except Exception as e:
        print(f"❌ Błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
