#!/usr/bin/env python3
"""
S04E04 - Drone Navigation Webhook z LangGraph
Multi-engine: openai, lmstudio, anything, gemini, claude
Automatyczne uruchomienie webhook API z ngrok exposure
"""
import argparse
import os
import sys
import re
import json
import requests
import subprocess
import signal
import time
import logging
import threading
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, List, Dict, Any, Tuple
from langgraph.graph import StateGraph, START, END

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Drone Navigation Webhook (multi-engine)")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
parser.add_argument("--port", type=int, default=3001, help="Port dla serwera webhook")
parser.add_argument("--skip-send", action="store_true", help="Nie wysyłaj URL do centrali automatycznie")
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
CENTRALA_API_KEY: str = os.getenv("CENTRALA_API_KEY")
REPORT_URL: str = os.getenv("REPORT_URL")

if not all([CENTRALA_API_KEY, REPORT_URL]):
    print("❌ Brak wymaganych zmiennych: CENTRALA_API_KEY, REPORT_URL", file=sys.stderr)
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

# 2. Inicjalizacja klienta LLM
def clean_llm_response(response: str) -> str:
    """Czyści odpowiedź LLM z tagów myślenia używanych przez modele lokalne"""
    # Usuń tagi <think>...</think> (nawet jeśli niedomknięte)
    response = re.sub(r'<think>.*?(?:</think>|$)', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Usuń tagi <thinking>...</thinking>
    response = re.sub(r'<thinking>.*?(?:</thinking>|$)', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Usuń tagi myślenia w innych formatach
    response = re.sub(r'<thought>.*?(?:</thought>|$)', '', response, flags=re.DOTALL | re.IGNORECASE)
    response = re.sub(r'\[THOUGHT\].*?(?:\[/THOUGHT\]|$)', '', response, flags=re.DOTALL | re.IGNORECASE)
    response = re.sub(r'\[THINKING\].*?(?:\[/THINKING\]|$)', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Usuń linie zaczynające się od myślenia
    lines = response.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not any(word in line.lower() for word in ['<think', 'okay', 'let me', 'breaking', 'analysis']):
            cleaned_lines.append(line)
    
    response = '\n'.join(cleaned_lines)
    
    # Usuń nadmiarowe białe znaki
    response = re.sub(r'\n\s*\n', '\n', response)
    response = re.sub(r'\s+', ' ', response)
    
    return response.strip()

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
        response = resp.choices[0].message.content.strip()
        
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
        response = resp.content[0].text.strip()
        
    elif ENGINE in {"lmstudio", "anything"}:
        from openai import OpenAI
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LMSTUDIO_API_KEY", "local") if ENGINE == "lmstudio" else os.getenv("ANYTHING_API_KEY", "local")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Optymalizacje dla modeli lokalnych z większymi tokenami
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise assistant. Give very short, direct answers. No thinking tags."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=200,  # Więcej tokenów na bezpieczeństwo
            timeout=15.0   # 15 sekund timeout
        )
        response = resp.choices[0].message.content.strip()
        
    elif ENGINE == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(
            [prompt],
            generation_config={"temperature": temperature, "max_output_tokens": 1000}
        )
        response = resp.text.strip()
    
    # Wyczyść odpowiedź z tagów myślenia (szczególnie dla modeli lokalnych)
    if ENGINE in {"lmstudio", "anything"}:
        response = clean_llm_response(response)
    
    return response

# 3. Mapa i logika drona (skopiowane z drone_navigation_langgraph.py)
MAP = [
    ["start", "łąka", "drzewo", "dom"],       # Wiersz 1 (góra)
    ["łąka", "wiatrak", "łąka", "łąka"],      # Wiersz 2
    ["łąka", "łąka", "skały", "drzewa"],      # Wiersz 3
    ["góry", "góry", "samochód", "jaskinia"]  # Wiersz 4 (dół)
]

# Stan dla LangGraph nawigacji drona
class NavigationState(TypedDict):
    instruction: str
    current_position: Tuple[int, int]
    movements: List[str]
    final_position: Tuple[int, int]
    description: str
    thinking: str

# Pydantic models dla API
class DroneInstruction(BaseModel):
    instruction: str

class DroneResponse(BaseModel):
    description: str
    _thinking: Optional[str] = None

# LangGraph nodes dla nawigacji drona
def parse_instruction_node(state: NavigationState) -> NavigationState:
    """Parsuje instrukcję na listę ruchów"""
    instruction = state["instruction"].lower()
    
    # Optymalizowany prompt dla modeli lokalnych
    if ENGINE in {"lmstudio", "anything"}:
        prompt = f"""Parse this Polish drone instruction to movement commands.

Instruction: "{instruction}"

Map: 4x4 grid. Drone moves: RIGHT, LEFT, UP, DOWN.

Key phrases:
- "maksymalnie w prawo" = RIGHT, RIGHT, RIGHT (3 moves to edge)
- "na sam dół" = DOWN, DOWN, DOWN (3 moves to edge)  
- "albo nie!", "czekaj" = CANCEL previous commands
- Follow order: "right then down" means RIGHT first, then DOWN

Return only comma-separated movements like: RIGHT, RIGHT, DOWN
If no movement: NONE

Answer:"""
    else:
        prompt = f"""Przeanalizuj poniższą instrukcję lotu drona i wypisz TYLKO listę ruchów.
Dron może się poruszać: PRAWO, LEWO, GÓRA, DÓŁ.
Mapa ma wymiary 4x4.

Instrukcja: "{instruction}"

Ważne wskazówki:
- "na maksa w prawo" = 3 razy w prawo (do końca mapy)
- "na sam dół" = 3 razy w dół (do końca mapy)
- "ile wlezie" = maksymalnie w danym kierunku
- Jeśli jest "albo nie!", "czekaj" lub podobne - to ANULUJE poprzednie ruchy
- Zwróć uwagę na kolejność: "w prawo, a później w dół" != "w dół, a później w prawo"

Zwróć TYLKO listę ruchów oddzielonych przecinkami, np: PRAWO, PRAWO, DÓŁ
Jeśli nie ma żadnych ruchów, zwróć: BRAK"""
    
    # Pomiar czasu dla debug
    start_time = time.time()
    logger.info(f"🤖 Wysyłam do {ENGINE}: {instruction[:50]}...")
    
    try:
        movements_str = call_llm(prompt)
    except Exception as e:
        logger.error(f"❌ Błąd LLM: {e}")
        # Fallback - spróbuj podstawowego parsowania
        movements_str = basic_instruction_parser(instruction)
        logger.info(f"🔄 Używam fallback parsera: {movements_str}")
    
    end_time = time.time()
    logger.info(f"⏱️  LLM odpowiedział w {end_time - start_time:.2f}s")
    logger.info(f"📤 Surowa odpowiedź: {repr(movements_str)}")
    
    state["thinking"] = f"LLM response ({end_time - start_time:.2f}s): {movements_str}"
    
    # Parsuj ruchy - obsługa różnych formatów odpowiedzi
    movements_str_upper = movements_str.strip().upper()
    if movements_str_upper in ["BRAK", "NONE", "NO MOVES", ""]:
        state["movements"] = []
        logger.info("🚫 Brak ruchów")
    else:
        # Zamień angielskie na polskie dla spójności
        movements_str = movements_str.replace("RIGHT", "PRAWO")
        movements_str = movements_str.replace("LEFT", "LEWO") 
        movements_str = movements_str.replace("UP", "GÓRA")
        movements_str = movements_str.replace("DOWN", "DÓŁ")
        
        movements = [m.strip().upper() for m in movements_str.split(",") if m.strip()]
        # Filtruj tylko poprawne ruchy
        valid_movements = [m for m in movements if m in ["PRAWO", "LEWO", "GÓRA", "DÓŁ"]]
        state["movements"] = valid_movements
        logger.info(f"🎯 Ruchy: {valid_movements}")
    
    return state

def basic_instruction_parser(instruction: str) -> str:
    """Podstawowy parser instrukcji jako fallback"""
    instruction = instruction.lower()
    
    # Sprawdź anulowanie
    if any(word in instruction for word in ["albo nie", "czekaj", "nie idziemy", "stop"]):
        # Sprawdź co jest po anulowaniu
        parts = re.split(r"albo nie|czekaj|nie idziemy|stop", instruction)
        if len(parts) > 1:
            instruction = parts[-1]  # Bierz ostatnią część
        else:
            return "NONE"
    
    movements = []
    
    # Maksymalne ruchy
    if "maksymalnie w prawo" in instruction or "na maksa w prawo" in instruction:
        movements.extend(["RIGHT", "RIGHT", "RIGHT"])
    elif "maksymalnie w lewo" in instruction or "na maksa w lewo" in instruction:
        movements.extend(["LEFT", "LEFT", "LEFT"])
    elif "na sam dół" in instruction or "maksymalnie w dół" in instruction:
        movements.extend(["DOWN", "DOWN", "DOWN"])
    elif "na sam góra" in instruction or "maksymalnie w górę" in instruction:
        movements.extend(["UP", "UP", "UP"])
    
    # Pojedyncze ruchy
    if "w prawo" in instruction and "maksymalnie" not in instruction:
        movements.append("RIGHT")
    if "w lewo" in instruction and "maksymalnie" not in instruction:
        movements.append("LEFT")
    if "w dół" in instruction and "sam dół" not in instruction:
        movements.append("DOWN")
    if "w górę" in instruction or "do góry" in instruction:
        movements.append("UP")
    
    return ", ".join(movements) if movements else "NONE"

def execute_movements_node(state: NavigationState) -> NavigationState:
    """Wykonuje ruchy i znajduje końcową pozycję"""
    # Zawsze startujemy z [1,1] (lewy górny róg)
    y, x = 1, 1
    
    thinking_log = f"Start: [{y},{x}] ({MAP[y-1][x-1]})\n"
    
    for movement in state["movements"]:
        old_y, old_x = y, x
        
        if movement == "PRAWO" and x < 4:
            x += 1
        elif movement == "LEWO" and x > 1:
            x -= 1
        elif movement == "DÓŁ" and y < 4:
            y += 1
        elif movement == "GÓRA" and y > 1:
            y -= 1
        
        thinking_log += f"{movement}: [{old_y},{old_x}] -> [{y},{x}] ({MAP[y-1][x-1]})\n"
    
    state["final_position"] = (y, x)
    state["thinking"] = state.get("thinking", "") + "\n" + thinking_log
    
    # Pobierz opis miejsca
    description = MAP[y-1][x-1]
    
    # Specjalne przypadki dla odpowiedzi
    if description == "drzewa":
        description = "drzewa"  # Już w liczbie mnogiej
    elif description == "góry":
        description = "góry"    # Już w liczbie mnogiej
    
    state["description"] = description
    
    return state

def build_navigation_graph():
    """Buduje graf nawigacji drona"""
    graph = StateGraph(NavigationState)
    
    # Dodaj nodes
    graph.add_node("parse_instruction", parse_instruction_node)
    graph.add_node("execute_movements", execute_movements_node)
    
    # Dodaj edges
    graph.add_edge(START, "parse_instruction")
    graph.add_edge("parse_instruction", "execute_movements")
    graph.add_edge("execute_movements", END)
    
    return graph.compile()

# 4. FastAPI app
app = FastAPI()
navigation_graph = build_navigation_graph()

@app.post("/", response_model=DroneResponse)
async def drone_navigation(data: DroneInstruction):
    """Endpoint przetwarzający instrukcje lotu drona"""
    try:
        logger.info(f"📥 Otrzymano instrukcję: {data.instruction}")
        
        # Uruchom graf nawigacji
        initial_state = NavigationState(
            instruction=data.instruction,
            current_position=(1, 1),
            movements=[],
            final_position=(1, 1),
            description="",
            thinking=""
        )
        
        result = navigation_graph.invoke(initial_state)
        
        logger.info(f"🤔 Thinking:\n{result['thinking']}")
        logger.info(f"📍 Końcowa pozycja: {result['final_position']}")
        logger.info(f"📝 Opis: {result['description']}")
        
        return DroneResponse(
            description=result["description"],
            _thinking=result["thinking"]
        )
        
    except Exception as e:
        logger.error(f"❌ Błąd: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "engine": ENGINE, "model": MODEL_NAME}

# 5. Typowanie stanu pipeline webhook
class WebhookState(TypedDict, total=False):
    server_process: Optional[subprocess.Popen]
    ngrok_process: Optional[subprocess.Popen]
    webhook_url: Optional[str]
    server_ready: bool
    port: int
    flag_found: bool
    result: Optional[str]

# 6. Funkcje pomocnicze
def check_ngrok_installed() -> bool:
    """Sprawdza czy ngrok jest zainstalowany"""
    try:
        result = subprocess.run(["ngrok", "version"], 
                               capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_port_available(port: int) -> bool:
    """Sprawdza czy port jest wolny"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Czeka aż serwer będzie gotowy"""
    import requests
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    
    return False

def get_ngrok_url() -> Optional[str]:
    """Pobiera publiczny URL z ngrok API"""
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=10)
        response.raise_for_status()
        tunnels = response.json()
        
        for tunnel in tunnels.get("tunnels", []):
            if tunnel.get("proto") == "https":
                return tunnel.get("public_url")
        
        # Fallback do pierwszego dostępnego tunelu
        if tunnels.get("tunnels"):
            return tunnels["tunnels"][0].get("public_url")
            
    except Exception as e:
        logger.error(f"❌ Błąd pobierania URL z ngrok: {e}")
    
    return None

def should_continue(state: WebhookState) -> str:
    """Decyduje czy kontynuować czy zakończyć na podstawie znalezienia flagi"""
    if state.get("flag_found", False) or args.skip_send:
        return "cleanup"
    else:
        return "wait_for_completion"

# 7. Nodes dla LangGraph webhook pipeline
def check_environment_node(state: WebhookState) -> WebhookState:
    """Sprawdza środowisko i zależności"""
    logger.info("🔍 Sprawdzam środowisko...")
    
    state["port"] = args.port
    
    # Sprawdź czy ngrok jest zainstalowany
    if not check_ngrok_installed():
        logger.error("❌ ngrok nie jest zainstalowany!")
        logger.info("📥 Instalacja:")
        logger.info("   - macOS: brew install ngrok") 
        logger.info("   - Linux: snap install ngrok")
        logger.info("   - Lub pobierz z: https://ngrok.com/download")
        raise RuntimeError("ngrok nie jest zainstalowany")
    
    logger.info("✅ ngrok jest zainstalowany")
    
    # Sprawdź czy port jest wolny
    if not check_port_available(state["port"]):
        logger.error(f"❌ Port {state['port']} jest zajęty!")
        raise RuntimeError(f"Port {state['port']} jest zajęty")
    
    logger.info(f"✅ Port {state['port']} jest dostępny")
    
    return state

def start_server_node(state: WebhookState) -> WebhookState:
    """Uruchamia serwer FastAPI w tle"""
    logger.info(f"🚀 Uruchamiam serwer na porcie {state['port']}...")
    
    # Uruchom serwer w osobnym procesie
    try:
        # Przygotuj środowisko dla subprocess
        env = os.environ.copy()
        env["LLM_ENGINE"] = ENGINE
        env["MODEL_NAME"] = MODEL_NAME
        
        # Uruchom uvicorn programowo w wątku
        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=state["port"], 
                       log_level="warning", access_log=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Czekaj aż serwer będzie gotowy
        if wait_for_server(state["port"]):
            logger.info("✅ Serwer jest gotowy!")
            state["server_ready"] = True
        else:
            logger.error("❌ Serwer nie odpowiada!")
            raise RuntimeError("Serwer nie uruchomił się poprawnie")
            
    except Exception as e:
        logger.error(f"❌ Błąd uruchamiania serwera: {e}")
        raise
    
    return state

def start_ngrok_node(state: WebhookState) -> WebhookState:
    """Uruchamia ngrok i pobiera publiczny URL"""
    logger.info("🔧 Uruchamiam ngrok...")
    
    try:
        # Uruchom ngrok w tle
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", str(state["port"])],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        state["ngrok_process"] = ngrok_process
        
        # Czekaj na uruchomienie ngrok
        time.sleep(5)
        
        # Sprawdź czy proces nadal działa
        if ngrok_process.poll() is not None:
            logger.error("❌ Ngrok zakończył się nieoczekiwanie!")
            raise RuntimeError("Ngrok nie uruchomił się poprawnie")
        
        # Pobierz publiczny URL
        webhook_url = get_ngrok_url()
        
        if webhook_url:
            logger.info(f"✅ Ngrok URL: {webhook_url}")
            state["webhook_url"] = webhook_url
        else:
            logger.error("❌ Nie udało się uzyskać URL z ngrok!")
            raise RuntimeError("Nie można uzyskać URL z ngrok")
            
    except Exception as e:
        logger.error(f"❌ Błąd uruchamiania ngrok: {e}")
        raise
    
    return state

def send_webhook_url_node(state: WebhookState) -> WebhookState:
    """Wysyła URL webhooka do centrali"""
    webhook_url = state.get("webhook_url")
    
    if not webhook_url:
        logger.error("❌ Brak URL webhooka!")
        return state
    
    if args.skip_send:
        logger.info("⏸️  Pomijam wysyłanie URL do centrali (--skip-send)")
        logger.info(f"📌 Twój webhook URL: {webhook_url}")
        logger.info("📌 Możesz go wysłać ręcznie przez:")
        logger.info(f"   curl -X POST {REPORT_URL} -H 'Content-Type: application/json' \\")
        logger.info(f"        -d '{{\"task\":\"webhook\",\"apikey\":\"{CENTRALA_API_KEY}\",\"answer\":\"{webhook_url}\"}}'")
        return state
    
    # Wyślij URL do centrali
    payload = {
        "task": "webhook",
        "apikey": CENTRALA_API_KEY,
        "answer": webhook_url
    }
    
    logger.info(f"📤 Wysyłam webhook URL: {webhook_url}")
    
    try:
        response = requests.post(REPORT_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"✅ Odpowiedź centrali: {result}")
        
        # Sprawdź czy jest flaga
        if "FLG" in str(result):
            logger.info(f"🏁 FLAGA: {result}")
            state["result"] = result.get("message", str(result))
            state["flag_found"] = True
            return state
        else:
            state["result"] = "URL wysłany pomyślnie"
            state["flag_found"] = False
            
    except Exception as e:
        logger.error(f"❌ Błąd wysyłania: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Szczegóły: {e.response.text}")
        state["flag_found"] = False
    
    return state

def wait_for_completion_node(state: WebhookState) -> WebhookState:
    """Czeka na zakończenie lub Ctrl+C (tylko jeśli brak flagi)"""
    logger.info("🔄 Serwer działa. Czekam na testy centrali...")
    logger.info("💡 Naciśnij Ctrl+C aby zakończyć ręcznie")
    
    try:
        # Czekaj w nieskończoność
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Otrzymano sygnał zakończenia...")
    
    return state

def cleanup_node(state: WebhookState) -> WebhookState:
    """Zatrzymuje wszystkie procesy"""
    logger.info("🧹 Sprzątam...")
    
    # Zatrzymaj ngrok
    ngrok_process = state.get("ngrok_process")
    if ngrok_process:
        try:
            ngrok_process.terminate()
            ngrok_process.wait(timeout=5)
            logger.info("✅ Ngrok zatrzymany")
        except subprocess.TimeoutExpired:
            ngrok_process.kill()
            logger.info("🔪 Ngrok zabity")
        except Exception as e:
            logger.warning(f"⚠️  Błąd zatrzymywania ngrok: {e}")
    
    # Serwer FastAPI zostanie zatrzymany automatycznie gdy główny proces się zakończy
    
    logger.info("✅ Sprzątanie zakończone")
    return state

def build_webhook_graph() -> Any:
    """Buduje graf LangGraph dla webhook pipeline"""
    graph = StateGraph(state_schema=WebhookState)
    
    # Dodaj nodes
    graph.add_node("check_environment", check_environment_node)
    graph.add_node("start_server", start_server_node)
    graph.add_node("start_ngrok", start_ngrok_node) 
    graph.add_node("send_webhook_url", send_webhook_url_node)
    graph.add_node("wait_for_completion", wait_for_completion_node)
    graph.add_node("cleanup", cleanup_node)
    
    # Dodaj edges
    graph.add_edge(START, "check_environment")
    graph.add_edge("check_environment", "start_server")
    graph.add_edge("start_server", "start_ngrok")
    graph.add_edge("start_ngrok", "send_webhook_url")
    
    # Conditional edge - jeśli flaga znaleziona lub skip-send, idź do cleanup
    graph.add_conditional_edges(
        "send_webhook_url",
        should_continue,
        {
            "wait_for_completion": "wait_for_completion",
            "cleanup": "cleanup"
        }
    )
    
    graph.add_edge("wait_for_completion", "cleanup")
    graph.add_edge("cleanup", END)
    
    return graph.compile()

def main() -> None:
    print("=== Zadanie 18: Drone Navigation Webhook ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔧 Model: {MODEL_NAME}")
    print(f"🌐 Port: {args.port}")
    print(f"📤 Pomiń wysyłanie: {'TAK' if args.skip_send else 'NIE'}")
    print("Startuje webhook pipeline...\n")
    
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
    
    try:
        graph = build_webhook_graph()
        result: WebhookState = graph.invoke({})
        
        if result.get("flag_found"):
            print(f"\n🏁 FLAGA ZNALEZIONA! {result.get('result', '')}")
        elif result.get("result"):
            print(f"\n🎉 Webhook pipeline zakończony: {result.get('result')}")
        else:
            print(f"\n✅ Pipeline zakończony")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Przerwano przez użytkownika")
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()