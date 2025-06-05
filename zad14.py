#!/usr/bin/env python3
"""
S03E05 - Znajdowanie najkrótszej ścieżki od Rafała do Barbary
Multi-engine: openai, lmstudio, anything, gemini, claude
Wykorzystuje bazę MySQL i Neo4j do analizy połączeń między osobami

### 1. **Prosty, czysty model danych**
```cypher
// Węzły: User z userId i name
CREATE (u:User {userId: 17, name: "Rafał"})

// Relacje: jednostronna KNOWS
CREATE (u1)-[:KNOWS]->(u2)
```

### 2. **Cypher robi całą robotę**
```cypher
MATCH (start:User {name: "Rafał"}), (end:User {name: "Barbara"}),
      path = shortestPath((start)-[:KNOWS*]->(end))
RETURN [n in nodes(path) | n.name] AS names
```
Neo4j automatycznie znajduje najkrótszą ścieżkę używając algorytmu BFS (Breadth-First Search).

### 3. **Dane z MySQL są kompletne**
- Tabela `connections` zawiera wszystkie potrzebne relacje
- Graf jest skierowany (user1_id → user2_id)
- Istnieje ścieżka: Rafał → ktoś → ktoś → Barbara

### 4. **LangGraph zapewnia porządek**
```
START → fetch_users → fetch_connections → create_graph → find_path → send_answer → END
```
Każdy krok jest niezależny i testowalny.

## Co się dzieje "pod maską":
1. **MySQL** → dostarcza surowe dane (users + connections)
2. **Neo4j** → buduje graf i znajduje optymalną ścieżkę
3. **Python** → tylko orkiestruje proces

**Fun fact**: Neo4j jest tak dobry w grafach, że `shortestPath()` to dla niego podstawowa operacja - jak `SELECT` dla SQL. 
"""
import argparse
import os
import sys
import requests
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, List, Dict, Any, Tuple
from langgraph.graph import StateGraph, START, END
from neo4j import GraphDatabase

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Znajdowanie najkrótszej ścieżki w grafie (multi-engine)")
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

if not all([APIDB_URL, REPORT_URL, CENTRALA_API_KEY]):
    print("❌ Brak wymaganych zmiennych: APIDB_URL, REPORT_URL, CENTRALA_API_KEY", file=sys.stderr)
    sys.exit(1)

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 2. Typowanie stanu pipeline
class PipelineState(TypedDict, total=False):
    users: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    shortest_path: List[str]
    result: str

# 3. Funkcje pomocnicze
def make_db_request(query: str) -> Optional[List[Dict[str, Any]]]:
    """Wykonuje zapytanie do API bazy danych"""
    payload = {
        "task": "database",
        "apikey": CENTRALA_API_KEY,
        "query": query
    }
    
    logger.info(f"📤 Wysyłam zapytanie SQL: {query}")
    
    try:
        response = requests.post(APIDB_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "reply" in result and result["reply"] is not None:
            logger.info(f"✅ Otrzymano {len(result['reply'])} rekordów")
            return result["reply"]
        else:
            logger.warning(f"⚠️  API zwróciło nieoczekiwaną odpowiedź: {result}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Błąd podczas wykonywania zapytania: {e}")
        return None

class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        """Czyści całą bazę danych"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("🧹 Wyczyszczono bazę Neo4j")
            
    def create_user_node(self, user_id: int, username: str):
        """Tworzy węzeł użytkownika"""
        with self.driver.session() as session:
            session.run(
                "CREATE (u:User {userId: $user_id, name: $username})",
                user_id=user_id, username=username
            )
            
    def create_connection(self, user1_id: int, user2_id: int):
        """Tworzy relację KNOWS między użytkownikami"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (u1:User {userId: $user1_id})
                MATCH (u2:User {userId: $user2_id})
                CREATE (u1)-[:KNOWS]->(u2)
                """,
                user1_id=user1_id, user2_id=user2_id
            )
            
    def find_shortest_path(self, start_name: str, end_name: str) -> Optional[List[str]]:
        """Znajduje najkrótszą ścieżkę między dwoma użytkownikami"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (start:User {name: $start_name}), (end:User {name: $end_name}),
                      path = shortestPath((start)-[:KNOWS*]->(end))
                RETURN [n in nodes(path) | n.name] AS names
                """,
                start_name=start_name, end_name=end_name
            )
            record = result.single()
            if record:
                return record["names"]
            return None

# 4. Nodes dla LangGraph
def fetch_users_node(state: PipelineState) -> PipelineState:
    """Pobiera listę użytkowników z bazy MySQL"""
    logger.info("📥 Pobieram użytkowników z bazy danych...")
    
    users = make_db_request("SELECT * FROM users")
    if users:
        state["users"] = users
        logger.info(f"✅ Pobrano {len(users)} użytkowników")
    else:
        logger.error("❌ Nie udało się pobrać użytkowników")
        state["users"] = []
    
    return state

def fetch_connections_node(state: PipelineState) -> PipelineState:
    """Pobiera listę połączeń z bazy MySQL"""
    logger.info("📥 Pobieram połączenia z bazy danych...")
    
    connections = make_db_request("SELECT * FROM connections")
    if connections:
        state["connections"] = connections
        logger.info(f"✅ Pobrano {len(connections)} połączeń")
    else:
        logger.error("❌ Nie udało się pobrać połączeń")
        state["connections"] = []
    
    return state

def create_graph_node(state: PipelineState) -> PipelineState:
    """Tworzy graf w Neo4j na podstawie danych z MySQL"""
    logger.info("🔨 Tworzę graf w Neo4j...")
    
    users = state.get("users", [])
    connections = state.get("connections", [])
    
    if not users or not connections:
        logger.error("❌ Brak danych do stworzenia grafu")
        return state
    
    neo4j = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Wyczyść bazę
        neo4j.clear_database()
        
        # Stwórz węzły użytkowników
        logger.info("📍 Tworzę węzły użytkowników...")
        for user in users:
            neo4j.create_user_node(user["id"], user["username"])
        
        # Stwórz relacje
        logger.info("🔗 Tworzę relacje między użytkownikami...")
        for conn in connections:
            neo4j.create_connection(conn["user1_id"], conn["user2_id"])
        
        logger.info("✅ Graf utworzony pomyślnie")
        
    except Exception as e:
        logger.error(f"❌ Błąd podczas tworzenia grafu: {e}")
    finally:
        neo4j.close()
    
    return state

def find_path_node(state: PipelineState) -> PipelineState:
    """Znajduje najkrótszą ścieżkę od Rafała do Barbary"""
    logger.info("🔍 Szukam najkrótszej ścieżki od Rafała do Barbary...")
    
    neo4j = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        path = neo4j.find_shortest_path("Rafał", "Barbara")
        
        if path:
            state["shortest_path"] = path
            state["result"] = ",".join(path)
            logger.info(f"✅ Znaleziono ścieżkę: {' -> '.join(path)}")
        else:
            logger.error("❌ Nie znaleziono ścieżki")
            state["shortest_path"] = []
            state["result"] = ""
            
    except Exception as e:
        logger.error(f"❌ Błąd podczas szukania ścieżki: {e}")
        state["shortest_path"] = []
        state["result"] = ""
    finally:
        neo4j.close()
    
    return state

def send_answer_node(state: PipelineState) -> PipelineState:
    """Wysyła odpowiedź do centrali"""
    logger.info("📡 Wysyłam odpowiedź do centrali...")
    
    result = state.get("result", "")
    
    if not result:
        logger.error("❌ Brak wyniku do wysłania")
        return state
    
    payload = {
        "task": "connections",
        "apikey": CENTRALA_API_KEY,
        "answer": result
    }
    
    logger.info(f"📤 Wysyłam: {payload}")
    
    try:
        response = requests.post(REPORT_URL, json=payload)
        response.raise_for_status()
        logger.info(f"✅ Odpowiedź centrali: {response.text}")
    except Exception as e:
        logger.error(f"❌ Błąd wysyłania: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"   Szczegóły: {e.response.text}")
    
    return state

def build_graph() -> Any:
    """Buduje graf LangGraph"""
    graph = StateGraph(state_schema=PipelineState)
    
    # Dodaj nodes
    graph.add_node("fetch_users", fetch_users_node)
    graph.add_node("fetch_connections", fetch_connections_node)
    graph.add_node("create_graph", create_graph_node)
    graph.add_node("find_path", find_path_node)
    graph.add_node("send_answer", send_answer_node)
    
    # Dodaj edges
    graph.add_edge(START, "fetch_users")
    graph.add_edge("fetch_users", "fetch_connections")
    graph.add_edge("fetch_connections", "create_graph")
    graph.add_edge("create_graph", "find_path")
    graph.add_edge("find_path", "send_answer")
    graph.add_edge("send_answer", END)
    
    return graph.compile()

def main() -> None:
    print("=== Zadanie 14: Znajdowanie najkrótszej ścieżki w grafie ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🌐 API URL: {APIDB_URL}")
    print(f"🔗 Neo4j URI: {NEO4J_URI}")
    print("Startuje pipeline...\n")
    
    try:
        # Sprawdź połączenie z Neo4j
        neo4j = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        neo4j.close()
        logger.info("✅ Połączono z Neo4j")
    except Exception as e:
        logger.error(f"❌ Nie można połączyć się z Neo4j: {e}")
        logger.error("Upewnij się, że Neo4j jest uruchomiony")
        sys.exit(1)
    
    try:
        graph = build_graph()
        result: PipelineState = graph.invoke({})
        
        if result.get("result"):
            print(f"\n🎉 Zadanie zakończone! Najkrótsza ścieżka: {result['result']}")
        else:
            print("\n❌ Nie udało się znaleźć ścieżki")
            
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
