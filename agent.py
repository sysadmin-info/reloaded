#!/usr/bin/env python3
import os
import sys
import re
import json
import subprocess
import platform
from dotenv import load_dotenv

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

def detect_shell_and_clean_env():
    """
    Wykrywa typ powłoki i czyści problematyczne zmienne środowiskowe
    które mogłyby konfliktować z ustawieniami z .env
    """
    # Zmienne które mogą powodować problemy
    problematic_vars = [
        "LLM_ENGINE", "MODEL_NAME", "ENGINE", 
        "OPENAI_MODEL", "CLAUDE_MODEL", "GEMINI_MODEL",
        "AI_MODEL", "LLM_MODEL_NAME"
    ]
    
    shell_type = "unknown"
    system = platform.system().lower()
    
    # Wykrywanie typu powłoki
    if system == "windows":
        if os.getenv("PSModulePath"):  # PowerShell
            shell_type = "powershell"
        else:  # CMD
            shell_type = "cmd"
    else:  # Linux/macOS
        shell = os.getenv("SHELL", "").lower()
        if "bash" in shell:
            shell_type = "bash"
        elif "zsh" in shell:
            shell_type = "zsh"
        elif "fish" in shell:
            shell_type = "fish"
        else:
            shell_type = "unix"
    
    print(f"🔍 Wykryto środowisko: {system} / {shell_type}")
    
    # Sprawdź problematyczne zmienne
    found_vars = []
    for var in problematic_vars:
        value = os.environ.get(var)
        if value:
            found_vars.append((var, value))
    
    if found_vars:
        print("⚠️  Znaleziono potencjalnie konfliktowe zmienne środowiskowe:")
        for var, value in found_vars:
            print(f"   {var} = {value}")
        
        # Zapytaj użytkownika czy wyczyścić
        try:
            response = input("❓ Wyczyścić te zmienne dla tej sesji? [y/N]: ").strip().lower()
            if response in {'y', 'yes', 'tak', 't'}:
                for var, _ in found_vars:
                    del os.environ[var]
                    print(f"🧹 Wyczyszczono: {var}")
                
                # Pokaż instrukcje jak wyczyścić na stałe
                print("\n💡 Aby wyczyścić zmienne na stałe:")
                if shell_type == "powershell":
                    for var, _ in found_vars:
                        print(f"   [Environment]::SetEnvironmentVariable('{var}', $null, 'User')")
                elif shell_type in ["bash", "zsh"]:
                    print("   Usuń odpowiednie linie z ~/.bashrc, ~/.zshrc, ~/.profile")
                elif shell_type == "cmd":
                    for var, _ in found_vars:
                        print(f"   setx {var} \"\"")
                print()
            else:
                print("🔄 Kontynuuję z obecnymi zmiennymi...")
        except (EOFError, KeyboardInterrupt):
            print("\n🔄 Kontynuuję z obecnymi zmiennymi...")
    else:
        print("✅ Brak konfliktowych zmiennych środowiskowych")

completed_tasks = set()
completed_secrets = set()  # NOWE: śledzenie ukończonych sekretów
current_engine = None
current_model = None

def _execute_task(task_key: str):
    key = str(task_key).strip().strip("'").strip('"')
    if key not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"}:
        return ("Niepoprawny numer zadania. Wybierz w zakresie 1-24.", False, False)
    script = f"zad{key}.py"
    if not os.path.exists(script):
        return (f"Plik {script} nie istnieje.", False, False)
    env = os.environ.copy()
    # POPRAWKA: przekazuj wybrany silnik i model do subprocess
    if current_engine:
        env["LLM_ENGINE"] = current_engine
    if current_model:
        env["MODEL_NAME"] = current_model
    if not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    env["PYTHONUTF8"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, script],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        output_full = result.stdout.rstrip()
        if output_full == "":
            output_full = "(Brak wyjścia)"
        flags = re.findall(r"\{\{FLG:.*?\}\}|FLG\{.*?\}", output_full)
        if flags:
            lines = output_full.splitlines()
            last_flag_index = 0
            for i, line in enumerate(lines):
                if "{{FLG:" in line or "FLG{" in line:
                    last_flag_index = i
            truncated_output = "\n".join(lines[:last_flag_index+1])
            return (flags, True, False)
        else:
            return (output_full, False, False)
    except subprocess.CalledProcessError as e:
        out_text = (e.stdout or "").rstrip()
        err_text = (e.stderr or "").rstrip()
        return ((out_text, err_text), False, True)

def _execute_secret(secret_key: str):
    """NOWE: Wykonuje sekretne zadanie secN.py"""
    key = str(secret_key).strip().strip("'").strip('"')
    if key not in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
        return ("Niepoprawny numer sekretu. Wybierz w zakresie 1-9git.", False, False)
    script = f"sec{key}.py"
    if not os.path.exists(script):
        return (f"Plik {script} nie istnieje.", False, False)
    env = os.environ.copy()
    # Przekazuj wybrany silnik i model do subprocess
    if current_engine:
        env["LLM_ENGINE"] = current_engine
    if current_model:
        env["MODEL_NAME"] = current_model
    if not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    env["PYTHONUTF8"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, script],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        output_full = result.stdout.rstrip()
        if output_full == "":
            output_full = "(Brak wyjścia)"
        flags = re.findall(r"\{\{FLG:.*?\}\}|FLG\{.*?\}", output_full)
        if flags:
            lines = output_full.splitlines()
            last_flag_index = 0
            for i, line in enumerate(lines):
                if "{{FLG:" in line or "FLG{" in line:
                    last_flag_index = i
            truncated_output = "\n".join(lines[:last_flag_index+1])
            return (flags, True, False)
        else:
            return (output_full, False, False)
    except subprocess.CalledProcessError as e:
        out_text = (e.stdout or "").rstrip()
        err_text = (e.stderr or "").rstrip()
        return ((out_text, err_text), False, True)

@tool
def run_task(task_key: str) -> str:
    """
    Uruchamia zadanie zadN.py (gdzie N to numer zadania) i zwraca wynik działania,
    w szczególności flagę w formacie {{FLG:...}}. Używany przez agenta LangChain.
    Obsługuje zadania 1-24.
    """
    key = str(task_key).strip().strip("'").strip('"')
    print(f"🔄 Uruchamiam zadanie {key}…")
    output, flag_found, error = _execute_task(key)
    if error:
        stdout_text, stderr_text = output if isinstance(output, tuple) else ("", "")
        print("🛑 Zadanie zakończone z błędem.")
        if stdout_text:
            print(f"🐞 STDOUT:\n{stdout_text}")
        if stderr_text:
            print(f"🐞 STDERR:\n{stderr_text}")
        log_entry = {
            "zadanie": key,
            "flagi": [],
            "debug_output": f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
        }
        _append_to_json_log(log_entry)
        return "🛑 Zadanie zakończone z błędem."
    if flag_found:
        completed_tasks.add(key)
        flags_list = output if isinstance(output, list) else [str(output)]
        if len(flags_list) > 1:
            flag_msg = f"🏁 Flagi znalezione: [{', '.join(flags_list)}] - kończę zadanie."
        else:
            flag_msg = f"🏁 Flaga znaleziona: {flags_list[0]} - kończę zadanie."
        print(flag_msg)
        log_entry = {
            "zadanie": key,
            "flagi": flags_list
        }
        _append_to_json_log(log_entry)
        return flag_msg
    return str(output)

@tool
def run_secret(secret_key: str) -> str:
    """
    NOWE: Uruchamia sekretne zadanie secN.py (gdzie N to numer sekretu) i zwraca wynik działania,
    w szczególności flagę w formacie {{FLG:...}}. Używany przez agenta LangChain.
    Obsługuje sekrety 1-9.
    """
    key = str(secret_key).strip().strip("'").strip('"')
    print(f"🔐 Uruchamiam sekret {key}…")
    output, flag_found, error = _execute_secret(key)
    if error:
        stdout_text, stderr_text = output if isinstance(output, tuple) else ("", "")
        print("🛑 Sekret zakończony z błędem.")
        if stdout_text:
            print(f"🐞 STDOUT:\n{stdout_text}")
        if stderr_text:
            print(f"🐞 STDERR:\n{stderr_text}")
        log_entry = {
            "sekret": key,
            "flagi": [],
            "debug_output": f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
        }
        _append_to_json_log(log_entry, log_file="secrets.json")  # Osobny plik dla sekretów
        return "🛑 Sekret zakończony z błędem."
    if flag_found:
        completed_secrets.add(key)
        flags_list = output if isinstance(output, list) else [str(output)]
        if len(flags_list) > 1:
            flag_msg = f"🏁 Flagi znalezione: [{', '.join(flags_list)}] - kończę sekret."
        else:
            flag_msg = f"🏁 Flaga znaleziona: {flags_list[0]} - kończę sekret."
        print(flag_msg)
        log_entry = {
            "sekret": key,
            "flagi": flags_list
        }
        _append_to_json_log(log_entry, log_file="secrets.json")  # Osobny plik dla sekretów
        return flag_msg
    return str(output)

@tool
def read_env(var: str) -> str:
    """
    Zwraca wartość zmiennej środowiskowej o nazwie var. Jeśli zmienna nie istnieje, zwraca '(niewartość)'.
    """
    key = str(var).strip().strip("'").strip('"')
    return os.getenv(key, "(niewartość)")

def _append_to_json_log(entry: dict, log_file="flags.json"):
    """ROZSZERZONE: Dodano parametr log_file dla różnych plików logów"""
    data = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
        except json.JSONDecodeError:
            data = []
    
    # Deduplikacja: nie zapisuj identycznego wpisu
    entry_type = "zadanie" if "zadanie" in entry else "sekret"
    if entry_type in entry and "flagi" in entry and entry["flagi"]:
        for d in data:
            if d.get(entry_type) == entry[entry_type] and set(d.get("flagi", [])) == set(entry["flagi"]):
                return
    data.append(entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    global current_engine, current_model
    
    # Sprawdź i wyczyść problematyczne zmienne środowiskowe
    detect_shell_and_clean_env()
    
    # Sprawdź aktualny ENGINE z .env (po ewentualnym wyczyszczeniu)
    current_engine = os.getenv("LLM_ENGINE", "").lower()
    print(f"🔍 Aktualny LLM_ENGINE z .env: '{current_engine}'")
    
    engine = ""
    while engine not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
        try:
            prompt_text = f"Wybierz silnik LLM [openai/lmstudio/anything/gemini/claude]"
            if current_engine in {"openai", "lmstudio", "anything", "gemini", "claude"}:
                prompt_text += f" (aktualny: {current_engine})"
            prompt_text += ": "
            engine = input(prompt_text).strip().lower()
            
            # Jeśli użytkownik nie wpisał nic, użyj aktualnego ENGINE z .env
            if not engine and current_engine in {"openai", "lmstudio", "anything", "gemini", "claude"}:
                engine = current_engine
                print(f"🔄 Używam silnika z .env: {engine}")
                
        except (EOFError, KeyboardInterrupt):
            print("\nKoniec.")
            return
        if engine not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
            print("⚠️ Nieznany wybór. Wpisz 'openai', 'lmstudio', 'anything', 'gemini' albo 'claude'.")

    print(f"🚀 Wybrany silnik: {engine}")

    # Ustaw globalne zmienne do przekazania subprocess
    current_engine = engine

    # Ustawienie MODEL_NAME na podstawie silnika
    if engine == "openai":
        model_name = os.getenv("MODEL_NAME_OPENAI")
        if not model_name:
            print("⚠️ Nie ustawiono MODEL_NAME_OPENAI w .env - przerwano działanie.")
            return
    elif engine == "claude":
        model_name = os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    elif engine == "lmstudio":
        model_name = os.getenv("MODEL_NAME_LM", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_ANY", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
    elif engine == "gemini":
        model_name = os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    else:  # anything
        model_name = os.getenv("MODEL_NAME_ANY", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_LM", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")

    print(f"🔧 Model: {model_name}")

    # Ustaw globalne zmienne do przekazania subprocess
    current_model = model_name

    # Debug informacji o zmiennych środowiskowych
    print("🔍 Debug - zmienne kluczowe:")
    if engine == "lmstudio":
        print(f"   LMSTUDIO_API_URL: {os.getenv('LMSTUDIO_API_URL')}")
        print(f"   LMSTUDIO_API_KEY: {os.getenv('LMSTUDIO_API_KEY')}")
    elif engine == "anything":
        print(f"   ANYTHING_API_URL: {os.getenv('ANYTHING_API_URL')}")
        print(f"   ANYTHING_API_KEY: {os.getenv('ANYTHING_API_KEY')}")
    elif engine == "claude":
        claude_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        print(f"   CLAUDE/ANTHROPIC_API_KEY: {'✅ ustawiony' if claude_key else '❌ brak'}")
    elif engine == "gemini":
        print(f"   GEMINI_API_KEY: {'✅ ustawiony' if os.getenv('GEMINI_API_KEY') else '❌ brak'}")
    elif engine == "openai":
        print(f"   OPENAI_API_KEY: {'✅ ustawiony' if os.getenv('OPENAI_API_KEY') else '❌ brak'}")
        print(f"   OPENAI_API_URL: {os.getenv('OPENAI_API_URL')}")

    # Sprawdzenie wymaganych zmiennych przed inicjalizacją
    if engine == "claude":
        if not (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            print("⚠️ Nie ustawiono CLAUDE_API_KEY ani ANTHROPIC_API_KEY w .env - przerwano działanie.")
            return
    elif engine == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️ Nie ustawiono GEMINI_API_KEY w .env - przerwano działanie.")
            return
    elif engine == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ Nie ustawiono OPENAI_API_KEY w .env - przerwano działanie.")
            return

    # Ustawienie MODEL_NAME w środowisku (również dla LangChain)
    os.environ["MODEL_NAME"] = model_name
    os.environ["LLM_ENGINE"] = engine

    # Inicjalizacja LLM - uproszczona wersja bez dodatkowych ustawień os.environ
    try:
        if engine == "claude":
            try:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    model_name=model_name,
                    anthropic_api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
                    temperature=0
                )
                print("✅ Claude LLM zainicjalizowany")
            except ImportError:
                print("⚠️ Brak langchain_anthropic. Zainstaluj: pip install langchain-anthropic")
                print("Lub użyj innego silnika.")
                return
                
        elif engine == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0
            )
            print("✅ Gemini LLM zainicjalizowany")
            
        elif engine == "lmstudio":
            lmstudio_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
            lmstudio_key = os.getenv("LMSTUDIO_API_KEY", "local")
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                openai_api_base=lmstudio_url,
                openai_api_key=lmstudio_key
            )
            print(f"✅ LMStudio LLM zainicjalizowany ({lmstudio_url})")
            
        elif engine == "anything":
            anything_url = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
            anything_key = os.getenv("ANYTHING_API_KEY", "local")
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                openai_api_base=anything_url,
                openai_api_key=anything_key
            )
            print(f"✅ Anything LLM zainicjalizowany ({anything_url})")
            
        else:  # openai
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                openai_api_base=os.getenv("OPENAI_API_URL"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            print("✅ OpenAI LLM zainicjalizowany")
            
    except Exception as e:
        print(f"❌ Błąd podczas inicjalizacji LLM: {e}")
        return

    # Konfiguracja grafu LangChain - ROZSZERZONE o run_secret
    tools = [run_task, run_secret, read_env]
    builder = StateGraph(AgentState)
    llm_with_tools = llm.bind_tools(tools)
    builder.add_node("agent", lambda state: {"messages": llm_with_tools.invoke(state["messages"])})
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    graph = builder.compile()
    
    print("🤖 Agent uruchomiony. Komendy: run_task N (1-24) | run_secret N (1-9) | read_env VAR | exit")
    print("=" * 60)
    
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKoniec.")
            break
        if not cmd:
            continue
        if cmd.lower() in {"exit", "quit"}:
            print("Wyłączam agenta.")
            break
            
        if cmd.lower().startswith("run_task"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                print("Niepoprawny numer zadania. Wybierz w zakresie 1-24.")
                continue
            task_arg = parts[1].strip()
            if (task_arg.startswith("'") and task_arg.endswith("'")) or (task_arg.startswith('"') and task_arg.endswith('"')):
                task_arg = task_arg[1:-1].strip()
            if task_arg not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"}:
                print("Niepoprawny numer zadania. Wybierz w zakresie 1-24.")
                continue
            task_id = task_arg
            print(f"🔄 Uruchamiam zadanie {task_id}…")
            output, flag_found, error = _execute_task(task_id)
            if error:
                stdout_text, stderr_text = output if isinstance(output, tuple) else ("", "")
                print("🛑 Zadanie zakończone z błędem.")
                if stdout_text:
                    print(f"🐞 STDOUT:\n{stdout_text}")
                if stderr_text:
                    print(f"🐞 STDERR:\n{stderr_text}")
                log_entry = {
                    "zadanie": task_id,
                    "flagi": [],
                    "debug_output": f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
                }
                _append_to_json_log(log_entry)
            elif flag_found:
                completed_tasks.add(task_id)
                flags_list = output if isinstance(output, list) else [str(output)]
                if len(flags_list) > 1:
                    print(f"🏁 Flagi znalezione: [{', '.join(flags_list)}] - kończę zadanie.")
                else:
                    print(f"🏁 Flaga znaleziona: {flags_list[0]} - kończę zadanie.")
                log_entry = {
                    "zadanie": task_id,
                    "flagi": flags_list
                }
                _append_to_json_log(log_entry)
            else:
                print(output)
            continue
            
        # Obsługa run_secret
        if cmd.lower().startswith("run_secret"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                print("Niepoprawny numer sekretu. Wybierz w zakresie 1-4.")
                continue
            secret_arg = parts[1].strip()
            if (secret_arg.startswith("'") and secret_arg.endswith("'")) or (secret_arg.startswith('"') and secret_arg.endswith('"')):
                secret_arg = secret_arg[1:-1].strip()
            if secret_arg not in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                print("Niepoprawny numer sekretu. Wybierz w zakresie 1-9.")
                continue
            secret_id = secret_arg
            print(f"🔐 Uruchamiam sekret {secret_id}…")
            output, flag_found, error = _execute_secret(secret_id)
            if error:
                stdout_text, stderr_text = output if isinstance(output, tuple) else ("", "")
                print("🛑 Sekret zakończony z błędem.")
                if stdout_text:
                    print(f"🐞 STDOUT:\n{stdout_text}")
                if stderr_text:
                    print(f"🐞 STDERR:\n{stderr_text}")
                log_entry = {
                    "sekret": secret_id,
                    "flagi": [],
                    "debug_output": f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
                }
                _append_to_json_log(log_entry, log_file="secrets.json")
            elif flag_found:
                completed_secrets.add(secret_id)
                flags_list = output if isinstance(output, list) else [str(output)]
                if len(flags_list) > 1:
                    print(f"🏁 Flagi znalezione: [{', '.join(flags_list)}] - kończę sekret.")
                else:
                    print(f"🏁 Flaga znaleziona: {flags_list[0]} - kończę sekret.")
                log_entry = {
                    "sekret": secret_id,
                    "flagi": flags_list
                }
                _append_to_json_log(log_entry, log_file="secrets.json")
            else:
                print(output)
            continue
            
        if cmd.lower().startswith("read_env"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                print("(niewartość)")
                continue
            var = parts[1].strip()
            if (var.startswith("'") and var.endswith("'")) or (var.startswith('"') and var.endswith('"')):
                var = var[1:-1].strip()
            value = os.getenv(var, "(niewartość)")
            print(value)
            continue
            
        print("Nieznana komenda. Użyj: run_task N (1-24), run_secret N (1-9), read_env VAR, lub exit.")

if __name__ == "__main__":
    main()