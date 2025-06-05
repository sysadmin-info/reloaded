# Drone Navigation Webhook

Rozwiązanie zadania "webhook" (Zadanie 18) - API interpretujące instrukcje lotu drona po mapie 4x4 z orkiestracją LangGraph.

## Opis zadania

1. **Mapa 4x4** - dron porusza się po siatce:
   ```
   [1,1] Start    [1,2] Łąka     [1,3] Drzewo    [1,4] Dom
   [2,1] Łąka     [2,2] Wiatrak  [2,3] Łąka      [2,4] Łąka
   [3,1] Łąka     [3,2] Łąka     [3,3] Skały     [3,4] Drzewa
   [4,1] Góry     [4,2] Góry     [4,3] Samochód  [4,4] Jaskinia
   ```

2. **API** przyjmuje instrukcje w języku naturalnym i zwraca co znajduje się na końcowej pozycji
3. **Dron zawsze startuje** z pozycji [1,1] (lewy górny róg)

## Szybki start

### 1. Przygotowanie środowiska

```bash
# Edytuj .env i ustaw klucze API
nano .env

# Instaluj zależności
pip install fastapi uvicorn python-dotenv langgraph langchain-core requests pydantic openai

# Dla innych silników:
pip install anthropic  # dla Claude
pip install google-generativeai  # dla Gemini
pip install langchain-anthropic  # dla Claude z agent.py
```

### 2. Instalacja ngrok

```bash
# macOS
brew install ngrok

# Ubuntu
snap install ngrok

# Lub pobierz z https://ngrok.com/download

# WSL / Linux bez snap
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok

# TOKEN - załóż konto i wygeneruj token: https://dashboard.ngrok.com/signup
ngrok config add-authtoken YOUR_TOKEN
```

### 3. Uruchomienie webhook

#### Opcja A: Bezpośrednie uruchomienie z obsługą wielu silników (zalecane)

```bash
# OpenAI (domyślne)
python zad18.py --engine openai

# Claude 
python zad18.py --engine claude

# Gemini
python zad18.py --engine gemini

# Lokalne modele (LMStudio)
python zad18.py --engine lmstudio

# Anything LLM
python zad18.py --engine anything
```

#### Opcja B: Przez agenta (dla czystego wyciągnięcia flagi)

```bash
# Uruchom agenta
python agent.py

# Wybierz silnik gdy zostaniesz poproszony, następnie uruchom:
> run_task 18

# Agent zwróci tylko flagę: {{FLG:}}
```

#### Opcja C: Niestandardowe opcje

```bash
# Użyj innego portu
python zad18.py --engine claude --port 3002

# Nie wysyłaj URL automatycznie (ręczne testowanie)
python zad18.py --engine openai --skip-send
```

### 4. Testowanie lokalne

```bash
# Testuj pojedynczą instrukcję gdy webhook działa
curl -X POST http://localhost:3001/ \
  -H "Content-Type: application/json" \
  -d '{"instruction":"idź w prawo i w dół"}'

# Oczekiwana odpowiedź:
# {"description":"wiatrak","_thinking":"..."}
```

## Architektura wielu silników

Webhook automatycznie wykrywa i używa określonego silnika LLM:

```python
# Obsługiwane silniki:
--engine openai     # Modele GPT przez OpenAI API
--engine claude     # Modele Claude przez Anthropic API  
--engine gemini     # Modele Gemini przez Google API
--engine lmstudio   # Lokalne modele przez LMStudio
--engine anything   # Lokalne modele przez Anything LLM
```

### Pipeline LangGraph

```python
[START]
   ↓
[check_environment] - Sprawdź ngrok i dostępność portu
   ↓
[start_server] - Uruchom webhook FastAPI w tle
   ↓
[start_ngrok] - Stwórz publiczny tunel i pobierz URL
   ↓
[send_webhook_url] - Wyślij URL do serwera centralnego
   ↓
[wait_for_completion] - Czekaj na testy (lub auto-wyjście po fladze)
   ↓
[cleanup] - Zatrzymaj wszystkie procesy
   ↓
[END]
```

### Nawigacja drona z LangGraph

```python
[START] 
   ↓
[parse_instruction] - LLM interpretuje instrukcję na ruchy
   ↓
[execute_movements] - Symuluje ruch drona po mapie
   ↓
[END] → zwraca opis miejsca
```

### Kluczowe komponenty:

1. **WebhookState** - stan pipeline webhook w LangGraph
2. **NavigationState** - stan przepływu nawigacji drona
3. **parse\_instruction\_node** - używa LLM do interpretacji instrukcji
4. **execute\_movements\_node** - wykonuje ruchy i sprawdza pozycję
5. **FastAPI endpoints** - `/` dla instrukcji, `/health` dla sprawdzenia

## 🔧 Konfiguracja

### Zmienne środowiskowe (.env)

```bash
# Serwer centralny
CENTRALA_API_KEY=twój-klucz
REPORT_URL=https://centrala.ag3nts.org/report

# Silnik LLM (opcjonalne - można ustawić przez --engine)
LLM_ENGINE=openai  # lub: claude, lmstudio, gemini, anything

# Klucze API (w zależności od silnika)
OPENAI_API_KEY=sk-...
CLAUDE_API_KEY=sk-ant-...
ANTHROPIC_API_KEY=sk-ant-...  # alternatywa do CLAUDE_API_KEY
GEMINI_API_KEY=AI...

# Modele dla każdego silnika
MODEL_NAME_OPENAI=gpt-4o-mini
MODEL_NAME_CLAUDE=claude-sonnet-4-20250514
MODEL_NAME_GEMINI=gemini-2.5-pro-latest
MODEL_NAME_LM=llama-3.3-70b-instruct
MODEL_NAME_ANY=llama-3.3-70b-instruct

# URL lokalnych modeli
LMSTUDIO_API_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=local
ANYTHING_API_URL=http://localhost:1234/v1
ANYTHING_API_KEY=local
```

## Opcje linii poleceń

```bash
python zad18.py [OPCJE]

Opcje:
  --engine {openai,claude,gemini,lmstudio,anything}
                        Backend LLM do użycia
  --port INTEGER        Port dla serwera webhook (domyślnie: 3001)
  --skip-send          Nie wysyłaj URL do serwera centralnego automatycznie
  --help               Pokaż tę wiadomość i wyjdź
```

## Przykłady instrukcji

Webhook obsługuje złożone instrukcje w języku naturalnym:

1. **Proste**: "idź w prawo", "poleć w dół"
2. **Złożone**: "na maksa w prawo, a później ile wlezie w dół"
3. **Z anulowaniem**: "leć w dół, albo nie! czekaj! w prawo"
4. **Wieloetapowe**: "w prawo, potem w dół, potem jeszcze raz w prawo"
5. **Angielskie przykłady**: "all the way right, then as far down as possible"

## Rozwiązywanie problemów

### "Port już zajęty"
```bash
# Znajdź proces
lsof -i :3001

# Zabij proces
kill -9 <PID>
```

### "Ngrok nie działa"
* Sprawdź czy masz konto ngrok (darmowe)
* Ustaw authtoken: `ngrok config add-authtoken <twój-token>`
* Sprawdź czy ngrok działa: `curl http://localhost:4040/api/tunnels`

### "LLM nie interpretuje poprawnie"
* Sprawdź model w .env (GPT-4 działa lepiej niż GPT-3.5)
* Zweryfikuj czy klucze API są ustawione poprawnie
* Dla lokalnych modeli, upewnij się że LMStudio/Anything działa

### "Silnik nie znaleziony"
* Zainstaluj wymagane pakiety:
  ```bash
  pip install anthropic  # dla Claude
  pip install google-generativeai  # dla Gemini
  pip install langchain-anthropic  # dla agent.py z Claude
  ```

## Struktura odpowiedzi

API zawsze zwraca:
```json
{
    "description": "nazwa_miejsca",
    "_thinking": "opcjonalne_logi_debugowania"
}
```

Gdzie `description` to maksymalnie 2 słowa opisujące miejsce docelowe.

## Uzyskanie flagi

### Metoda 1: Bezpośrednie uruchomienie
```bash
python zad18.py --engine claude
# Webhook automatycznie zakończy się po otrzymaniu: {{FLG:}}
```

### Metoda 2: Przez agenta (czyste wyjście)
```bash
python agent.py
> run_task 18
# Zwraca tylko: 🏁 Flaga znaleziona: {{FLG:}} - kończę zadanie.
```

## Przepływ procesu

1. **Sprawdzenie środowiska** - Weryfikacja instalacji ngrok i dostępności portu
2. **Uruchomienie serwera** - Start serwera webhook FastAPI w wątku tła
3. **Tworzenie tunelu** - Uruchomienie ngrok i pobranie publicznego URL HTTPS
4. **Rejestracja** - Wysłanie URL webhook do serwera centralnego
5. **Faza testowania** - Serwer centralny testuje webhook różnymi instrukcjami
6. **Otrzymanie flagi** - Jeśli wszystkie testy przejdą, otrzymaj flagę i auto-wyjście
7. **Sprzątanie** - Zatrzymanie wszystkich procesów i czyste wyjście

Cały proces jest zautomatyzowany i zakończy się bez ingerencji użytkownika po uruchomieniu.