#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadanie 8: Odczyt nazwy flagi z zaszyfrowanego ciągu i dekodowanie ciągu przy pomocy LLM
Obsługiwane silniki: openai, lmstudio (Anything LLM), gemini
"""
import os
import re
import sys
from dotenv import load_dotenv

# 1. Wczytanie konfiguracji
ENGINE = os.getenv("LLM_ENGINE", "openai").lower()

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

# 5. Konfiguracja klienta LLM
if ENGINE in {"openai","lmstudio","anything"}:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))
    MODEL = os.getenv("MODEL_NAME_OPENAI", os.getenv("MODEL_NAME","gpt-4o-mini"))
elif ENGINE == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    MODEL = os.getenv("MODEL_NAME_GEMINI","gemini-2.5-pro-latest")
else:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
    sys.exit(1)

# 6. Funkcja wywołania LLM
def call_llm(sys_p, usr_p):
    if ENGINE in {"openai","lmstudio","anything"}:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":sys_p},{"role":"user","content":usr_p}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    else:
        model_llm = genai.GenerativeModel(MODEL)
        resp = model_llm.generate_content([sys_p, usr_p], generation_config={"temperature":0.0,"max_output_tokens":64})
        return resp.text.strip()

# 7. Odczyt odpowiedzi i ekstrakcja flagi — obsługa <think>…</think>
raw_name = call_llm(system_prompt, user_prompt)

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
print(flag)
