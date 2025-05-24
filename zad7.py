#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zad7.py - Multi-engine obraz robota: OpenAI (DALL-E), LM Studio, Anything, Gemini (brak wsparcia), Claude (brak wsparcia), ComfyUI

- LM Studio i Anything uruchamiają generowanie przez lokalny backend ComfyUI
- Dla ComfyUI wymagany workflow w formacie API (Save (API Format) w UI)
- Prompt pobierany dynamicznie z Centrali
- Claude NIE OBSŁUGUJE generowania obrazów - dodano komunikat informacyjny
- BEZPOŚREDNIA INTEGRACJA Claude (jak zad1.py i zad2.py)
- POPRAWKA: Lepsze wykrywanie silnika z agent.py

Wymagane zmienne środowiskowe (w .env):
- OPENAI_API_KEY, OPENAI_API_URL (opcjonalnie), MODEL_NAME_IMAGE
- REPORT_URL, CENTRALA_API_KEY, LOCAL_SD_API_URL
"""
import argparse
import os
import sys
import re
import requests
import json
import uuid
import glob
import time
import platform
from dotenv import load_dotenv

load_dotenv(override=True)

ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"


def banner(title: str) -> str:
    if sys.stdout.isatty():
        return f"{ANSI_GREEN}=== {title} ==={ANSI_RESET}"
    return f"=== {title} ==="


def extract_flag(text: str) -> str:
    m = re.search(r"\{\{FLG:[^}]+\}\}|FLG\{[^}]+\}", text)
    return m.group(0) if m else ""


def convert_workflow_to_api(workflow_path):
    with open(workflow_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "nodes" in data:
        nodes_api = {str(node["id"]): node for node in data["nodes"]}
    else:
        nodes_api = {str(k): v for k, v in data.items()}
    return nodes_api


def generate_with_comfyui(prompt, workflow_path, local_sd_api_url, output_dir):
    workflow_api = convert_workflow_to_api(workflow_path)
    # Znajdź node CLIPTextEncode
    prompt_node_id = None
    for node_id, node in workflow_api.items():
        if node.get("class_type") == "CLIPTextEncode":
            prompt_node_id = node_id
            break
    if not prompt_node_id:
        prompt_node_id = [k for k, v in workflow_api.items() if v.get("class_type") == "CLIPTextEncode"][0]
    workflow_api[prompt_node_id]["inputs"]["text"] = prompt
    # Ustaw filename_prefix dla SaveImage
    for node in workflow_api.values():
        if node.get("class_type") == "SaveImage":
            node["inputs"]["filename_prefix"] = "robot"

    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow_api, "client_id": client_id}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    try:
        response = requests.post(f"{local_sd_api_url}/prompt", json=payload)
        response.raise_for_status()
        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            print("❌ Brak prompt_id w odpowiedzi ComfyUI.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"❌ Błąd wysyłania żądania do ComfyUI: {e}", file=sys.stderr)
        sys.exit(1)

    # Poll na output_dir
    print("Czekam na wygenerowanie pliku w katalogu output...")
    last_time = time.time()
    timeout = 180
    found = None
    while time.time() - last_time < timeout:
        files = glob.glob(os.path.join(output_dir, "robot_*.png"))
        if files:
            latest_file = max(files, key=os.path.getctime)
            if os.path.getctime(latest_file) > last_time:
                found = latest_file
                break
        time.sleep(1.5)
    if found:
        print(f"Znaleziono plik: {found}")
        return found

    print("❌ Nie udało się znaleźć wygenerowanego obrazka po 3 minutach", file=sys.stderr)
    sys.exit(1)


def check_comfyui_api(url):
    try:
        requests.get(url, timeout=5)
    except Exception:
        print("❌ Brak działającego backendu ComfyUI. Zainstaluj ComfyUI z https://www.comfy.org/download", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generowanie obrazu robota (openai/lmstudio/anything/gemini/claude/comfyui)")
    parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude", "comfyui"],
                        help="Backend LLM do użycia")
    parser.add_argument("--workflow", default="robot.json",
                        help="Ścieżka do pliku workflow ComfyUI (API format)")
    parser.add_argument("--output-dir", default=None,
                        help="Katalog output ComfyUI (domyślnie wykrywany)")
    args = parser.parse_args()

    # POPRAWKA: Lepsze wykrywanie silnika (jak w poprawionych zad1.py-zad6.py)
    ENGINE = None
    if args.engine:
        ENGINE = args.engine.lower()
    elif os.getenv("LLM_ENGINE"):
        ENGINE = os.getenv("LLM_ENGINE").lower()
    else:
        # Próbuj wykryć silnik na podstawie ustawionych zmiennych MODEL_NAME
        model_name = os.getenv("MODEL_NAME", "")
        if "claude" in model_name.lower():
            ENGINE = "claude"
        elif "gemini" in model_name.lower():
            ENGINE = "gemini"
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            ENGINE = "openai"
        else:
            # Sprawdź które API keys są dostępne
            if os.getenv("OPENAI_API_KEY"):
                ENGINE = "openai"  # DALL-E jest najlepszy do obrazów
            elif os.getenv("LOCAL_SD_API_URL"):
                ENGINE = "comfyui"  # lokalna alternatywa
            else:
                ENGINE = "openai"  # domyślnie (generowanie obrazów wymaga chmury)

    if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude", "comfyui"}:
        print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
        sys.exit(1)

    print(f"🔄 ENGINE wykryty: {ENGINE}")

    # Sprawdzenie czy wybrany silnik obsługuje generowanie obrazów
    if ENGINE in {"claude", "gemini"}:
        print(f"❌ {ENGINE} nie obsługuje generowania obrazów.", file=sys.stderr)
        print("💡 Użyj --engine openai (DALL-E) lub comfyui (lokalne) dla generowania obrazów.", file=sys.stderr)
        sys.exit(1)

    WORKFLOW = args.workflow

    # Ustal OUTPUT_DIR dynamicznie jeśli nie podano
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        home = os.path.expanduser("~")
        if platform.system() == "Windows":
            OUTPUT_DIR = os.path.join(home, "ComfyUI", "output")
        else:
            OUTPUT_DIR = os.path.join(home, "ComfyUI", "output")

    # Zmienne środowiskowe
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    REPORT_URL = os.getenv("REPORT_URL", "")
    CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY", "")
    LOCAL_SD_API_URL = os.getenv("LOCAL_SD_API_URL", "http://localhost:8074")
    MODEL_NAME_IMAGE = os.getenv("MODEL_NAME_IMAGE", "dall-e-3")  # Używaj dedykowanej zmiennej do obrazów

    if not CENTRALA_API_KEY or not REPORT_URL:
        print("❌ Brak ustawienia CENTRALA_API_KEY lub REPORT_URL w .env", file=sys.stderr)
        sys.exit(1)

    # Sprawdzenie wymaganych API keys/zasobów
    if ENGINE == "openai" and not OPENAI_API_KEY:
        print("❌ Brak OPENAI_API_KEY dla DALL-E", file=sys.stderr)
        sys.exit(1)
    elif ENGINE in {"lmstudio", "anything", "comfyui"}:
        check_comfyui_api(LOCAL_SD_API_URL)

    print(f"✅ Zainicjalizowano silnik: {ENGINE} z obsługą generowania obrazów")

    # 1. Pobranie opisu robota
    ROBOT_URL = os.getenv("ROBOT_URL")
    if ROBOT_URL is None:
        print("❌ Brak ROBOT_URL w .env", file=sys.stderr)
        sys.exit(1)
        
    print(banner("Opis robota (dynamiczny)"))
    try:
        resp = requests.get(ROBOT_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        description = data.get("description", "").strip()
    except Exception as e:
        print(f"❌ Błąd pobierania lub parsowania JSON z {ROBOT_URL}: {e}", file=sys.stderr)
        sys.exit(1)
    if not description:
        print("❌ Brak pola 'description' w odpowiedzi.", file=sys.stderr)
        sys.exit(1)
    print(description)
    prompt = description

    print(f"🚀 Używam silnika: {ENGINE}")
    image_url = None

    if ENGINE == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ Zainstaluj openai (pip install openai)", file=sys.stderr)
            sys.exit(1)
        img_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)
        print(banner("Generowanie obrazu (OpenAI/DALL-E)"))
        print(f"[DEBUG] Używam modelu: {MODEL_NAME_IMAGE}")
        try:
            resp_img = img_client.images.generate(
                model=MODEL_NAME_IMAGE,
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            image_url = resp_img.data[0].url
            print(f"[📊 DALL-E - brak szczegółów tokenów]")
            print(f"[💰 DALL-E - sprawdź koszty w OpenAI Dashboard]")
        except Exception as e:
            print(f"❌ Błąd generowania obrazu przez OpenAI: {e}", file=sys.stderr)
            sys.exit(1)
    elif ENGINE in {"lmstudio", "anything", "comfyui"}:
        print(banner("Generowanie obrazu (ComfyUI API)"))
        print(f"[DEBUG] Workflow: {WORKFLOW}")
        print(f"[DEBUG] Output dir: {OUTPUT_DIR}")
        image_url = generate_with_comfyui(prompt, WORKFLOW, LOCAL_SD_API_URL, OUTPUT_DIR)
        print(f"[📊 ComfyUI - model lokalny]")
        print(f"[💰 ComfyUI - brak kosztów]")
    elif ENGINE == "claude":
        # Bezpośrednia integracja Claude (jak w zad1.py i zad2.py)
        print(banner("❌ Claude (Anthropic) nie obsługuje generowania obrazów!"))
        print("Claude jest modelem tekstowym i nie może generować obrazów.")
        print("Dostępne opcje:")
        print("1. Użyj OpenAI (DALL-E): ustaw LLM_ENGINE=openai w .env")
        print("2. Użyj ComfyUI (local): ustaw LLM_ENGINE=comfyui w .env")
        print("3. Uruchom ponownie agent.py i wybierz inny silnik")
        print("\nClaude świetnie sprawdza się w zadaniach tekstowych i analizy obrazów (vision),")
        print("ale nie może tworzyć nowych obrazów.")
        sys.exit(2)
    elif ENGINE == "gemini":
        print(banner("❌ Gemini (Google Generative AI) nie obsługuje generowania obrazów!"))
        print("Gemini w obecnej wersji nie obsługuje generowania obrazów.")
        print("Użyj OpenAI (DALL-E) lub ComfyUI (local)")
        sys.exit(2)
    else:
        print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
        sys.exit(1)

    if not image_url:
        print("❌ Nie uzyskano URL/ścieżki obrazu.", file=sys.stderr)
        sys.exit(1)
    print(banner("Ścieżka wygenerowanego obrazka"))
    print(image_url)

    # 4. Wysłanie do Centrali
    print(banner("Wysyłanie do Centrali"))
    payload = {"task": "robotid", "apikey": CENTRALA_API_KEY, "answer": image_url}
    try:
        r = requests.post(REPORT_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Błąd wysyłania do Centrali: {e}", file=sys.stderr)
        sys.exit(1)
    if r.status_code == 403:
        print("❌ Centralna odrzuciła (HTTP 403).", file=sys.stderr)
        sys.exit(1)
    if not r.ok:
        print(f"❌ Błąd HTTP {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)
    resp_text = r.text.strip()
    flag = extract_flag(resp_text)
    if flag:
        print(banner("Flaga"))
        print(flag)
    else:
        print("Brak flagi w odpowiedzi. Tekst serwera:", resp_text)


if __name__ == "__main__":
    main()