#!/usr/bin/env python3
"""
S04E03 - Uniwersalny mechanizm przeszukiwania stron internetowych
Zadanie: Odpowiedz na pytania centrali przeszukując stronę firmy SoftoAI
"""
import argparse
import os
import sys
import requests
import json
import re
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, List, Dict, Any, Set
from langgraph.graph import StateGraph, START, END
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import html2text

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1. Konfiguracja i wykrywanie silnika
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Uniwersalny agent przeszukujący strony (multi-engine)")
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
REPORT_URL: str = os.getenv("REPORT_URL")
CENTRALA_API_KEY: str = os.getenv("CENTRALA_API_KEY")
QUESTIONS_URL: str = os.getenv("SOFTO_QUESTIONS_URL")
SOFTO_URL: str = os.getenv("SOFTO_URL")

if not all([REPORT_URL, CENTRALA_API_KEY, QUESTIONS_URL, SOFTO_URL]):
    print("❌ Brak wymaganych zmiennych: REPORT_URL, CENTRALA_API_KEY, SOFTO_QUESTIONS_URL, SOFTO_URL", file=sys.stderr)
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
class SearchState(TypedDict):
    """Stan wyszukiwania dla pojedynczego pytania"""
    question_id: str
    question_text: str
    current_url: str
    visited_urls: Set[str]
    search_depth: int
    answer: Optional[str]
    found: bool

class PipelineState(TypedDict, total=False):
    questions: Dict[str, str]
    search_states: Dict[str, SearchState]
    answers: Dict[str, str]
    result: Optional[str]

# 4. Funkcje pomocnicze
def fetch_questions() -> Optional[Dict[str, str]]:
    """Pobiera pytania z API centrali"""
    logger.info(f"📥 Pobieranie pytań z centrali...")
    
    try:
        response = requests.get(QUESTIONS_URL)
        response.raise_for_status()
        questions = response.json()
        logger.info(f"✅ Pobrano {len(questions)} pytań")
        return questions
    except Exception as e:
        logger.error(f"❌ Błąd pobierania pytań: {e}")
        return None

def fetch_webpage(url: str) -> Optional[str]:
    """Pobiera zawartość strony internetowej"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"❌ Błąd pobierania strony {url}: {e}")
        return None

def html_to_markdown(html_content: str) -> str:
    """Konwertuje HTML na Markdown"""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0  # Nie zawijaj tekstu
    return h.handle(html_content)

def extract_links(html_content: str, base_url: str) -> List[Dict[str, str]]:
    """Ekstraktuje linki ze strony HTML"""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        
        # Pomiń kotwice i JavaScript
        if href.startswith('#') or href.startswith('javascript:'):
            continue
        
        # Utwórz pełny URL
        full_url = urljoin(base_url, href)
        
        # Sprawdź czy link jest w domenie softo.ag3nts.org
        parsed = urlparse(full_url)
        if parsed.netloc and 'softo.ag3nts.org' not in parsed.netloc:
            continue
        
        links.append({
            'url': full_url,
            'text': text or href
        })
    
    return links

def check_for_answer(content: str, question: str) -> Optional[str]:
    """Sprawdza czy strona zawiera odpowiedź na pytanie"""
    prompt = f"""Przeanalizuj poniższą treść strony i odpowiedz czy zawiera ona odpowiedź na pytanie.

Pytanie: {question}

Treść strony:
{content[:4000]}  # Ograniczenie długości

Instrukcje:
1. Jeśli strona zawiera konkretną odpowiedź na pytanie, zwróć TYLKO tę odpowiedź (bez dodatkowych wyjaśnień).
2. Jeśli strona nie zawiera odpowiedzi, zwróć dokładnie: "BRAK ODPOWIEDZI"

Odpowiedź:"""
    
    response = call_llm(prompt)
    
    if "BRAK ODPOWIEDZI" in response.upper():
        return None
    
    # Oczyść odpowiedź z niepotrzebnych elementów
    answer = response.strip()
    
    # Usuń typowe prefiksy
    prefixes_to_remove = [
        "Odpowiedź na pytanie to:",
        "Odpowiedź:",
        "Odpowiedź to:",
        "Adres e-mail to:",
        "Email to:",
        "Adres email:",
    ]
    
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    return answer

def select_best_link(links: List[Dict[str, str]], question: str, visited_urls: Set[str]) -> Optional[str]:
    """Wybiera najlepszy link do dalszego przeszukiwania"""
    # Filtruj odwiedzone linki
    unvisited_links = [link for link in links if link['url'] not in visited_urls]
    
    if not unvisited_links:
        return None
    
    # Przygotuj listę linków dla LLM
    links_text = "\n".join([f"{i+1}. {link['text']} -> {link['url']}" for i, link in enumerate(unvisited_links)])
    
    prompt = f"""Analizuję stronę w poszukiwaniu odpowiedzi na pytanie:
{question}

Dostępne linki:
{links_text}

Który link najprawdopodobniej doprowadzi do odpowiedzi? Rozważ:
1. Treść linku i jego związek z pytaniem
2. Typową strukturę stron (np. "Kontakt" dla adresów email, "O nas" dla informacji o firmie)
3. Unikaj linków które wyglądają na pułapki (np. loop, endless, czescizamienne)

Zwróć TYLKO numer linku (np. "3") bez żadnych dodatkowych wyjaśnień."""
    
    response = call_llm(prompt)
    
    # Wyciągnij numer
    try:
        number = int(re.search(r'\d+', response).group())
        if 1 <= number <= len(unvisited_links):
            return unvisited_links[number - 1]['url']
    except:
        pass
    
    # Jeśli nie udało się, zwróć pierwszy nieodwiedzony
    return unvisited_links[0]['url'] if unvisited_links else None

# 5. Nodes dla LangGraph
def fetch_questions_node(state: PipelineState) -> PipelineState:
    """Pobiera pytania z centrali"""
    questions = fetch_questions()
    if not questions:
        logger.error("❌ Nie udało się pobrać pytań")
        return state
    
    state["questions"] = questions
    
    # Inicjalizuj stany wyszukiwania dla każdego pytania
    search_states = {}
    for q_id, q_text in questions.items():
        search_states[q_id] = SearchState(
            question_id=q_id,
            question_text=q_text,
            current_url=SOFTO_URL,
            visited_urls=set(),
            search_depth=0,
            answer=None,
            found=False
        )
    
    state["search_states"] = search_states
    state["answers"] = {}
    
    return state

def search_answers_node(state: PipelineState) -> PipelineState:
    """Przeszukuje strony w poszukiwaniu odpowiedzi"""
    search_states = state.get("search_states", {})
    answers = state.get("answers", {})
    
    max_depth = 10  # Maksymalna głębokość przeszukiwania
    
    for q_id, search_state in search_states.items():
        if search_state["found"]:
            continue
        
        logger.info(f"\n🔍 Szukam odpowiedzi na pytanie {q_id}: {search_state['question_text']}")
        
        while search_state["search_depth"] < max_depth and not search_state["found"]:
            current_url = search_state["current_url"]
            
            # Sprawdź czy już odwiedziliśmy tę stronę
            if current_url in search_state["visited_urls"]:
                logger.warning(f"⚠️  Strona już odwiedzona: {current_url}")
                break
            
            logger.info(f"📄 Pobieram stronę: {current_url}")
            
            # Pobierz stronę
            html_content = fetch_webpage(current_url)
            if not html_content:
                break
            
            # Oznacz jako odwiedzoną
            search_state["visited_urls"].add(current_url)
            
            # Konwertuj na Markdown
            markdown_content = html_to_markdown(html_content)
            
            # Sprawdź czy jest odpowiedź
            answer = check_for_answer(markdown_content, search_state["question_text"])
            
            if answer:
                logger.info(f"✅ Znaleziono odpowiedź: {answer}")
                search_state["answer"] = answer
                search_state["found"] = True
                answers[q_id] = answer
                break
            
            # Jeśli nie ma odpowiedzi, wybierz następny link
            logger.info("❌ Brak odpowiedzi na tej stronie")
            
            # Wyciągnij linki
            links = extract_links(html_content, current_url)
            logger.info(f"🔗 Znaleziono {len(links)} linków")
            
            # Wybierz najlepszy link
            next_url = select_best_link(links, search_state["question_text"], search_state["visited_urls"])
            
            if not next_url:
                logger.warning("⚠️  Brak więcej linków do sprawdzenia")
                break
            
            logger.info(f"➡️  Przechodzę do: {next_url}")
            search_state["current_url"] = next_url
            search_state["search_depth"] += 1
            
            # Krótka przerwa aby nie przeciążać serwera
            time.sleep(0.5)
        
        if not search_state["found"]:
            logger.warning(f"⚠️  Nie znaleziono odpowiedzi na pytanie {q_id}")
    
    state["answers"] = answers
    return state

def send_answers_node(state: PipelineState) -> PipelineState:
    """Wysyła odpowiedzi do centrali"""
    answers = state.get("answers", {})
    
    if not answers:
        logger.error("❌ Brak odpowiedzi do wysłania")
        return state
    
    # Przygotuj payload
    payload = {
        "task": "softo",
        "apikey": CENTRALA_API_KEY,
        "answer": answers
    }
    
    logger.info(f"📤 Wysyłam odpowiedzi do centrali...")
    logger.info(f"Odpowiedzi: {json.dumps(answers, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(REPORT_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"✅ Odpowiedź centrali: {result}")
        
        # Sprawdź czy jest flaga
        if "FLG" in str(result):
            print(f"🏁 {result.get('message', result)}")
            state["result"] = result.get("message", str(result))
        
    except Exception as e:
        logger.error(f"❌ Błąd wysyłania: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Szczegóły: {e.response.text}")
    
    return state

def build_graph() -> Any:
    """Buduje graf LangGraph"""
    graph = StateGraph(state_schema=PipelineState)
    
    # Dodaj nodes
    graph.add_node("fetch_questions", fetch_questions_node)
    graph.add_node("search_answers", search_answers_node)
    graph.add_node("send_answers", send_answers_node)
    
    # Dodaj edges
    graph.add_edge(START, "fetch_questions")
    graph.add_edge("fetch_questions", "search_answers")
    graph.add_edge("search_answers", "send_answers")
    graph.add_edge("send_answers", END)
    
    return graph.compile()

def main() -> None:
    print("=== Uniwersalny Agent Przeszukujący Strony ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔧 Model: {MODEL_NAME}")
    print(f"🌐 Strona do przeszukania: {SOFTO_URL}")
    print("Startuje pipeline...\n")
    
    try:
        graph = build_graph()
        result: PipelineState = graph.invoke({})
        
        if result.get("result"):
            print(f"\n🎉 Zadanie zakończone!")
        else:
            print("\n✅ Proces zakończony")
            
            # Pokaż podsumowanie
            answers = result.get("answers", {})
            if answers:
                print(f"\n📊 Znaleziono {len(answers)} odpowiedzi:")
                for q_id, answer in answers.items():
                    print(f"   {q_id}: {answer}")
            else:
                print("\n❌ Nie znaleziono żadnych odpowiedzi")
                
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
