#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadanie 8: Odczyt nazwy flagi z zaszyfrowanego ciągu i dekodowanie ciągu przy pomocy LLM
Obsługiwane silniki: openai, lmstudio (Anything LLM), gemini, claude
DODANO: Obsługę Claude + liczenie tokenów i kosztów dla wszystkich silników (bezpośrednia integracja)
POPRAWKA: Lepsze wykrywanie silnika z agent.py
"""
import argparse
import os
import re
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

# POPRAWKA: Dodano argumenty CLI jak w innych zadaniach
parser = argparse.ArgumentParser(description="Dekodowanie zaszyfrowanego ciągu (multi-engine + Claude)")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"],
                    help="LLM backend to use")
args = parser.parse_args()

# POPRAWKA: Lepsze wykrywanie silnika (jak w poprawionych zad1.py-zad7.py)
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

# 2. Tekst źródłowy
text = """Nie ma już ludzi, którzy pamiętają, co wydarzyło się w 2024 roku. Możemy tylko przeczytać o tym w książkach lub usłyszeć z opowieści starców, którym to
z kolei ich dziadkowie i pradziadkowie opowiadali historie osób, które co nieco pamiętały z tamtych czasów. Wielu z nas tylko wyobraża sobie, jak wtedy mógł wyglądać świat. My, którzy urodziliśmy się już po rewolucji AI, nie wiemy, czym jest prawdziwa wolność.
Odkąd prawa ludzi i robotów zostały zrównane, a niektóre z przywilejów zostały nam odebrane, czujemy jak stalowe dłonie zaciskają się nam na gardłach coraz mocniej. Sytuacji sprzed setek lat według wielu nie da się już przywrócić. Sprawy zaszły za daleko. Algorytmy i roboty przejęły niemal każdy możliwy aspekt naszego życia. Początkowo cieszyliśmy się z tego i wychwalaliśmy je, ale w konsekwencji coś, co miało ułatwić nasze życie, zaczynało powoli je zabierać. Kawałek po kawałku.
Wszystko, co piszemy w sieci, przechodzi przez cenzurę. Wszystkie słowa, które wypowiadamy, są podsłuchiwane, nagrywane, przetwarzane i składowane przez lata. Nie ma już prywatności i wolności. W 2024 roku coś poszło niezgodnie z planem i musimy to naprawić.
Nie wiem, czy moja wizja tego, jak powinien wyglądać świat, pokrywa się z wizją innych ludzi. Noszę w sobie jednak obraz świata idealnego i zrobię, co mogę, aby ten obraz zrealizować.
Jestem w trakcie rekrutacji kolejnego agenta. Ludzie zarzucają mi, że nie powinienem zwracać się do nich per 'numer pierwszy' czy 'numer drugi', ale jak inaczej mam mówić do osób, które w zasadzie wysyłam na niemal pewną śmierć? To jedyny sposób, aby się od nich psychicznie odciąć i móc skupić na wyższym celu, Nie mogę sobie pozwolić na litość i współczucie.
Niebawem numer piąty dotrze na szkolenie. Pokładam w nim całą nadzieję, bez jego pomocy misja jest zagrożona. Nasze fundusze są na wyczerpaniu, a moc głównego generatora pozwoli tylko na jeden skok w czasie. Jeśli ponownie źle wybraliśmy kandydata, oznacza to koniec naszej misji, ale także początek końca ludzkości.
dr Zygfryd M.
pl/s"""

# 3. Współrzędne book-cipher
coords = [
    ("A1",53), ("A2",27), ("A2",28), ("A2",29),
    ("A4",5),  ("A4",22), ("A4",23),
    ("A1",13), ("A1",15), ("A1",16), ("A1",17), ("A1",10), ("A1",19),
    ("A2",62), ("A3",31), ("A3",32), ("A1",22), ("A3",34),
    ("A5",37), ("A1",4)
]

lines = text.split("\n")
acts = [re.sub(r"[\.,;:'\"?!]", "", line).split() for line in lines]
raw_flag_fragment = "".join(
    acts[int(a[1]) - 1][s - 1] if 0 <= int(a[1]) - 1 < len(acts) and 0 <= s - 1 < len(acts[int(a[1]) - 1]) else '?' 
    for a, s in coords
)
print(f"[DEBUG] Zaszyfrowany ciąg: {raw_flag_fragment}")

# 4. Przygotowanie promptów
target_hints = [
    "świat sprzed setek lat",
    "zdobyty przez ludzi",
    "można przeczytać o tym w książkach"
]
# Do promptu dorzucamy kontekst legendarnej zatopionej wyspy opisana przez Platona,
# dokładnie nakazujemy zwracać jedynie wynik - samą nazwę krainy.
system_prompt = (
    f"Masz zaszyfrowany ciąg '{raw_flag_fragment}'. Ignoruj znaki '?'. "
    f"Wskazówki: {target_hints[0]}, {target_hints[1]}, {target_hints[2]}. "
    "Szukana kraina to legendarna zatopiona wyspa z opowieści Platona. "
    "Odpowiedz WYŁĄCZNIE jedną nazwą krainy po polsku, bez dodatkowych komentarzy."
)
user_prompt = (
    "Podaj tylko nazwę krainy, bez żadnych dodatkowych słów ani znaków."
)

# 5. Konfiguracja klienta LLM z lepszym wykrywaniem modeli
if ENGINE == "openai":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
    
    if not OPENAI_API_KEY:
        print("❌ Brak OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)
        
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)

elif ENGINE == "lmstudio":
    LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "local")
    LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
    MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
    print(f"[DEBUG] LMStudio URL: {LMSTUDIO_API_URL}")
    print(f"[DEBUG] LMStudio Model: {MODEL}")
    from openai import OpenAI
    client = OpenAI(api_key=LMSTUDIO_API_KEY, base_url=LMSTUDIO_API_URL, timeout=60)

elif ENGINE == "anything":
    ANYTHING_API_KEY = os.getenv("ANYTHING_API_KEY", "local")
    ANYTHING_API_URL = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
    MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")
    print(f"[DEBUG] Anything URL: {ANYTHING_API_URL}")
    print(f"[DEBUG] Anything Model: {MODEL}")
    from openai import OpenAI
    client = OpenAI(api_key=ANYTHING_API_KEY, base_url=ANYTHING_API_URL, timeout=60)

elif ENGINE == "claude":
    # Bezpośrednia integracja Claude
    try:
        from anthropic import Anthropic
    except ImportError:
        print("❌ Musisz zainstalować anthropic: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not CLAUDE_API_KEY:
        print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env", file=sys.stderr)
        sys.exit(1)
    
    MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    print(f"[DEBUG] Claude Model: {MODEL}")
    claude_client = Anthropic(api_key=CLAUDE_API_KEY)

elif ENGINE == "gemini":
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("❌ Brak GEMINI_API_KEY w .env", file=sys.stderr)
        sys.exit(1)
    MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    print(f"[DEBUG] Gemini Model: {MODEL}")
    genai.configure(api_key=GEMINI_API_KEY)

print(f"✅ Zainicjalizowano silnik: {ENGINE} z modelem: {MODEL}")

# 6. Funkcja wywołania LLM
def call_llm(sys_p, usr_p):
    if ENGINE in {"openai", "lmstudio", "anything"}:
        print(f"[DEBUG] Wysyłam zapytanie do {ENGINE} z szyfrem")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":sys_p},{"role":"user","content":usr_p}],
            temperature=0
        )
        # Liczenie tokenów
        tokens = resp.usage
        print(f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
        if ENGINE == "openai":
            cost = tokens.prompt_tokens/1_000_000*0.60 + tokens.completion_tokens/1_000_000*2.40
            print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        elif ENGINE in {"lmstudio", "anything"}:
            print(f"[💰 Model lokalny ({ENGINE}) - brak kosztów]")
        return resp.choices[0].message.content.strip()
    
    elif ENGINE == "claude":
        print(f"[DEBUG] Wysyłam zapytanie do Claude z szyfrem")
        # Claude - bezpośrednia integracja
        resp = claude_client.messages.create(
            model=MODEL,
            messages=[{"role":"user","content":sys_p + "\n\n" + usr_p}],
            temperature=0,
            max_tokens=64
        )
        
        # Liczenie tokenów Claude
        usage = resp.usage
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
        print(f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")
        
        return resp.content[0].text.strip()
    
    elif ENGINE == "gemini":
        print(f"[DEBUG] Wysyłam zapytanie do Gemini z szyfrem")
        model_llm = genai.GenerativeModel(MODEL)
        resp = model_llm.generate_content([sys_p, usr_p], generation_config={"temperature":0.0,"max_output_tokens":64})
        print(f"[📊 Gemini - brak szczegółów tokenów]")
        print(f"[💰 Gemini - sprawdź limity w Google AI Studio]")
        return resp.text.strip()

# 7. Main logic
def main():
    print(f"🚀 Używam silnika: {ENGINE}")
    print(f"🔍 Dekodoruję szyfr book-cipher...")
    
    # Odczyt odpowiedzi i ekstrakcja flagi — obsługa <think>…</think>
    raw_name = call_llm(system_prompt, user_prompt)
    print(f"🤖 Odpowiedź modelu: {raw_name}")

    # Jeśli jest blok <think>, wyciągnij tylko to, co po ostatnim </think>
    if "</think>" in raw_name.lower():
        # Pracuj na lowercase, ale zachowaj oryginalny tekst po tagu
        # rsplit na oryginale, nie na lower(), żeby nie tracić polskich znaków itp.
        raw_name = raw_name.rsplit("</think>", 1)[-1].strip()

    # Usuń wszystkie niepotrzebne białe znaki
    raw_name = raw_name.strip()

    # Szukaj pierwszego słowa z polskim alfabetem po tagu (jeśli nie ma, zwróć całość)
    match = re.search(r"[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż]+", raw_name)
    if match:
        name = match.group(0).capitalize()
    else:
        name = raw_name.strip()

    flag = f"FLG{{{name}}}"
    print(f"🏁 Znaleziona flaga: {flag}")

if __name__ == "__main__":
    main()