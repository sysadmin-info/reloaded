#!/usr/bin/env python3
import os
import sys
import re
import json
import subprocess
from dotenv import load_dotenv

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

completed_tasks = set()

def _execute_task(task_key: str):
    key = str(task_key).strip().strip("'").strip('"')
    if key not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
        return ("Niepoprawny numer zadania. Wybierz w zakresie 1-10.", False, False)
    script = f"zad{key}.py"
    if not os.path.exists(script):
        return (f"Plik {script} nie istnieje.", False, False)
    env = os.environ.copy()
    if not env.get("MODEL_NAME"):
        env["MODEL_NAME"] = env.get("MODEL_NAME_LM") or env.get("MODEL_NAME_ANY") or env.get("MODEL_NAME_GEMINI") or env.get("MODEL_NAME_OPENAI") or env.get("MODEL_NAME_CLAUDE") or ""
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
    Obsługuje zadania 1-10.
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
def read_env(var: str) -> str:
    """
    Zwraca wartość zmiennej środowiskowej o nazwie var. Jeśli zmienna nie istnieje, zwraca '(niewartość)'.
    """
    key = str(var).strip().strip("'").strip('"')
    return os.getenv(key, "(niewartość)")

def _append_to_json_log(entry: dict):
    log_file = "flags.json"
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
    if "zadanie" in entry and "flagi" in entry and entry["flagi"]:
        for d in data:
            if d.get("zadanie") == entry["zadanie"] and set(d.get("flagi", [])) == set(entry["flagi"]):
                return
    data.append(entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    engine = ""
    # DODANO: claude do wyboru silników
    while engine not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
        try:
            engine = input("Wybierz silnik LLM [openai/lmstudio/anything/gemini/claude]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nKoniec.")
            return
        if engine not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
            print("⚠️ Nieznany wybór. Wpisz 'openai', 'lmstudio', 'anything', 'gemini' albo 'claude'.")

    if engine == "openai":
        os.environ["LLM_ENGINE"] = "openai"
        os.environ["OPENAI_API_URL"] = "https://api.openai.com/v1"
    elif engine == "claude":
        # DODANO: obsługa Claude
        os.environ["LLM_ENGINE"] = "claude"
        if not os.getenv("CLAUDE_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            print("⚠️ Nie ustawiono CLAUDE_API_KEY ani ANTHROPIC_API_KEY w .env - przerwano działanie.")
            return
    elif engine == "gemini":
        os.environ["LLM_ENGINE"] = "gemini"
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️ Nie ustawiono GEMINI_API_KEY w .env - przerwano działanie.")
            return
        os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    else:
        os.environ["LLM_ENGINE"] = engine
        os.environ["OPENAI_API_URL"] = "http://localhost:1234/v1"
        os.environ["OPENAI_API_KEY"] = "local"

    if engine == "openai":
        model_name = os.getenv("MODEL_NAME_OPENAI")
        if not model_name:
            print("⚠️ Nie ustawiono MODEL_NAME_OPENAI w .env - przerwano działanie.")
            return
    elif engine == "claude":
        # DODANO: model dla Claude
        model_name = os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    elif engine == "lmstudio":
        model_name = os.getenv("MODEL_NAME_LM", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_ANY", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
    elif engine == "gemini":
        model_name = os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    else:
        model_name = os.getenv("MODEL_NAME_ANY", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_LM", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
    os.environ["MODEL_NAME"] = model_name

    # DODANO: inicjalizacja LLM dla Claude
    if engine == "claude":
        # Claude nie jest bezpośrednio obsługiwany przez LangChain w tej wersji
        # Używamy ChatOpenAI z custom base_url (jeśli masz proxy Claude->OpenAI)
        # Alternatywnie możemy użyć ChatAnthropic jeśli dostępne
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model_name=model_name,
                anthropic_api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
                temperature=0
            )
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
    else:
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_base=os.getenv("OPENAI_API_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    tools = [run_task, read_env]
    builder = StateGraph(AgentState)
    llm_with_tools = llm.bind_tools(tools)
    builder.add_node("agent", lambda state: {"messages": llm_with_tools.invoke(state["messages"])})
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    graph = builder.compile()
    print("Agent uruchomiony. Komendy: run_task N (1-10) | read_env VAR | exit")
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
                print("Niepoprawny numer zadania. Wybierz w zakresie 1-10.")
                continue
            task_arg = parts[1].strip()
            if (task_arg.startswith("'") and task_arg.endswith("'")) or (task_arg.startswith('"') and task_arg.endswith('"')):
                task_arg = task_arg[1:-1].strip()
            if task_arg not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
                print("Niepoprawny numer zadania. Wybierz w zakresie 1-10.")
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
        print("Nieznana komenda. Użyj: run_task N (1-10), read_env VAR, lub exit.")

if __name__ == "__main__":
    main()