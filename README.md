# AI Devs 3 Reloaded - zadania w Python. Orkiestrator: LangGraph.

### English version below.

Zadania AI Devs 3 Reloaded w Pythonie, LangGraph i innych narzędziach

* **Sterowniki GPU:** W systemie Windows zainstaluj sterownik NVIDIA zgodny z WSL2 i CUDA (najlepiej najnowszy z serii 525+). Upewnij się, że Windows wykrywa GPU, a WSL ma do niego dostęp (polecenie `nvidia-smi` w WSL2 powinno wyświetlić RTX 3060).
* **Zasilacz i eGPU:** Zasilacz o mocy 650 W jest wystarczający dla RTX 3060. Ponieważ nie ma NVLink, nie możemy rozdzielić obliczeń na kilka kart – cały model musi zmieścić się na jednej karcie (lub częściowo w RAM). W praktyce ustawienie **gpu_layers=99** (ponad 98%) umieszcza większość sieci na GPU, co wymaga kwantyzacji i optymalizacji pamięci, by zmieścić się w limicie 12 GB VRAM.

# WSL2

Zainstaluj Windows Subsystem for Linux przez PowerShell:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
````

W systemie Windows włącz funkcję **WSL2** (PowerShell: `wsl --install`) i dodaj dystrybucję **Ubuntu 22.04 / 24.04 lub nowszą** ze Sklepu Microsoft. Tak, instalacja NVIDIA CUDA nie działa na Debian 12 😂. Następnie w WSL:

```bash
sudo apt update
sudo apt install -y build-essential dkms cmake git python3 python3-pip nvidia-cuda-toolkit nvidia-cuda-dev libcurl4-openssl-dev curl jq unzip zipalign
```

W PowerShell wykonaj:

```powershell
wsl --shutdown
```

Po ponownym uruchomieniu spróbuj w WSL polecenia `nvidia-smi`; jeśli zobaczysz listę procesorów, GPU jest gotowe.

Aby uruchomić zadania:

Zmień nazwę `_env` na `.env`, zmodyfikuj go, dodaj brakujące wartości i zapisz.

Wykonaj poniższe polecenia:

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-pol ffmpeg
git clone https://github.com/sysadmin-info/reloaded.git
cd reloaded
mkdir venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 agent.py
```

Wybierz dostawcę zdalnego lub lokalnego, a następnie wpisz `run_task N`, gdzie `N` to numer zadania, np. 1, 2, 3 itd.

# PowerShell – Windows 11

Jeśli wolisz PowerShell, poniżej znajduje się przewodnik:

Zainstaluj Python 3.11.9 z tej strony: [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)

Wykonaj poniższe polecenia:

```powershell
py -3.11 -m venv "dev-venv"
\dev-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install openai-whisper requests opencv-python pytesseract langdetect langchain-core langchain-openai langchain-google-genai langgraph dotenv bs4 google.generativeai
```

Zainstaluj Chocolatey, postępując zgodnie z tym przewodnikiem: [Chocolatey](https://docs.chocolatey.org/en-us/choco/setup/), a następnie zainstaluj ffmpeg i tesseract.
Uruchom w PowerShell jako administrator:

```powershell
choco install ffmpeg tesseract
```

### LM Studio i Anything LLM

Pobierz i zainstaluj:

[LM Studio](https://lmstudio.ai/)
[Anything LLM](https://anythingllm.com/)

Najpierw pobierz kilka modeli w LM Studio, a następnie naucz się go używać. W Anything LLM ustaw backend na LM Studio z rozwijanej listy (automatycznie rozpozna dostępne modele).

Następnie uruchom:

```powershell
cd reloaded
source venv/bin/activate
python3 agent.py
```

# Jak działa agent.py?

To, co widzisz (tylko flaga na wyjściu przy uruchamianiu przez `agent.py`), to **zamierzone zachowanie** i efekt przemyślanego designu.

### Dlaczego wyjście jest "czyste" (tylko flaga)?

1. **`agent.py`** uruchamia zadania (`zadN.py`) jako **subprocess** z przekierowanym wyjściem (stdout) - czyli pobiera całość wyjścia skryptu, a następnie:

   * Parsuje tylko flagę (lub flagi) z wyjścia (`{{FLG:...}}`).
   * **Wszystkie inne komunikaty i logi są ignorowane** (chyba że pojawił się błąd).
   * Na ekranie pojawia się tylko końcowy komunikat z flagą (`🏁 Flaga znaleziona: ... - kończę zadanie.`), a nie cała "gadająca" konsola z `zadN.py`.
   * Zobacz fragment funkcji `_execute_task` oraz `run_task` w `agent.py`:

     ```python
     output_full = result.stdout.rstrip()
     flags = re.findall(r"\{\{FLG:.*?\}\}|FLG\{.*?\}", output_full)
     if flags:
         # ... drukuje tylko flagi
     ```
   * **Wszystkie komunikaty typu "⚠️ Tylko OpenAI obsługuje embeddingi..." pojawią się tylko przy uruchomieniu `zad12.py` bezpośrednio**, a nie przez agenta.

2. **Dzięki temu możesz spokojnie wrzucać ile chcesz printów/debugów do zad12.py** (nawet ostrzeżenia, info, warningi), a agent pokaże tylko flagę, bez śmieciowego logowania na konsoli.

---

### **Jak to działa?**

* **Pośrednie uruchomienie przez agenta**: pokazuje tylko flagę.
* **Bezpośrednie uruchomienie** (`python zadN.py`): pokazuje wszystkie printy i logi.

---

### **Co zrobić, jeśli chcesz zmienić to zachowanie?**

1. **Chcesz widzieć więcej logów przez agenta?**
   – Możesz łatwo zmienić fragment `_execute_task` w `agent.py`, by wyświetlał także `stdout` jeśli nie ma flag, albo nawet zawsze (dodatkowa linia printu z całością).

2. **Chcesz zostawić jak jest?**
   – Aktualna forma jest **idealna do zawodów CTF/AI-Devs** - tylko flagi na konsoli, czysty output, automatyczna rejestracja flag.

---

**Podsumowując:**
To zamierzone, przemyślane zachowanie.
Chcesz zobaczyć "pełne gadanie" - uruchamiasz bezpośrednio.
Chcesz czystość i flaga-only - uruchamiasz przez agenta.
Wszystko masz skonfigurowane **wzorowo**!

---

Jakbyś chciał „hybrydę” — np. dodatkowe info na wyjściu przez agenta - daj znać, podam gotową zmianę do `agent.py`.
Możesz też wrzucać `print(..., file=sys.stderr)` i przekierować je, jeśli chcesz np. debug na stderr a flagę na stdout (do własnych automatyzacji).

# Zadanie 12 - zad12.py

## Uruchomienie Qdrant w Docker:

```bash
# Podstawowe uruchomienie Qdrant (podobnie jak Twój istniejący)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant

# Z persystencją danych (zalecane)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Restart istniejącego kontenera
docker restart hardcore_jackson
```

## Aktualizacja kodu zad12.py dla Docker Qdrant:

Zmień linię inicjalizacji Qdrant z:
```python
qdrant_client = QdrantClient(":memory:")
```

na:
```python
qdrant_client = QdrantClient("localhost", port=6333)
```

## Docker Desktop w WSL2 z Ubuntu 24.04.2:

1. **Zainstaluj Docker Desktop na Windows**:
   - Pobierz z: https://www.docker.com/products/docker-desktop
   - Podczas instalacji zaznacz "Use WSL 2 instead of Hyper-V"

2. **Włącz integrację z WSL2**:
   - Otwórz Docker Desktop
   - Settings → Resources → WSL Integration
   - Włącz "Enable integration with my default WSL distro"
   - Zaznacz swoją dystrybucję Ubuntu

3. **Sprawdź w WSL2**:
   ```bash
   # Powinno działać bez sudo
   docker version
   docker ps
   ```

4. **Jeśli masz problemy z uprawnieniami**:
   ```bash
   # Dodaj użytkownika do grupy docker
   sudo usermod -aG docker $USER
   
   # Wyloguj się i zaloguj ponownie lub:
   newgrp docker
   ```

## Przydatne komendy Qdrant:

```bash
# Sprawdź logi
docker logs hardcore_jackson

# Sprawdź stan kolekcji przez API
curl http://localhost:6333/collections

# Powinno zwrócić coś w stylu:
# {"result":{"collections":[]},"status":"ok","time":0.000123}

# Dashboard Qdrant (jeśli port 6334 jest otwarty)
# Otwórz w przeglądarce: http://localhost:6333/dashboard
```

#### Aktywuj swoje środowisko wirtualne (jeśli używasz)

```bash
source venv/bin/activate  # lub jak masz nazwane
```

#### Zainstaluj qdrant-client

```bash
pip install qdrant-client
```

#### w `.env` dodaj to:

```bash
# Domyślnie używa localhost:6333
QDRANT_URL=localhost
QDRANT_PORT=6333

# Lub dla in-memory (bez Dockera)
# QDRANT_URL=:memory:
```

##### Podsumowanie opcji konfiguracji:

### 1. **In-memory** (najprostsze, do testów)
```bash
# W .env
QDRANT_HOST=:memory:
```
- ✅ Nie wymaga Dockera
- ✅ Działa od razu
- ❌ Dane znikają po zakończeniu

### 2. **Docker lokalny** (WSL2, Linux, macOS)
```bash
# Uruchom Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# W .env (lub zostaw domyślne)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```
- ✅ Persystencja danych
- ✅ Pełna funkcjonalność
- ⚠️ Wymaga Dockera

### 3. **Docker zdalny** (VM, serwer)
```bash
# W .env
QDRANT_HOST=192.168.1.100  # IP twojego serwera
QDRANT_PORT=6333
```

### 4. **Qdrant Cloud**
```bash
# W .env
QDRANT_URL=https://xyz.qdrant.io
QDRANT_API_KEY=your_qdrant_key
```

Kod automatycznie wykryje konfigurację i pokaże odpowiednie komunikaty. Jeśli połączenie z Dockerem nie działa, dostaniesz pomocne wskazówki.

# Zadanie 14 - zad14.py i sekret 4.

Zainstaluj Docker, zainstaluj kontener z neo4j

```bash
docker run -d   --name neo4j   -p 7474:7474 -p 7687:7687   -v $HOME/neo4j/data:/data   -v $HOME/neo4j/logs:/logs   -v $HOME/neo4j/import:/import   -v $HOME/neo4j/plugins:/plugins   -e NEO4J_AUTH=neo4j/YourStrongPassword123   neo4j:5
```

Zainstaluj neo4j i YouTube downloader

```bash
pip install neo4j yt-dlp
```

# Zadanie 15 i sekret 5.

Spróbuj uruchomić za pomocą:

```bash
python zad15.py --engine openai
```

Jeśli nadal będą problemy z tokenami, możesz użyć wersji z małymi obrazkami:

```bash
python zad15.py --engine openai --use-small
```

Jeśli chcesz tylko flagę, uruchom za pomocą:

```bash
python agent.py
```

wybierz silnik i wpisz: `run_task 15`

Dla LM Studio i Anything LLM pobierz [qwen2.5-vl-7b](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

# Zadanie 16

Jeśli masz poprawny URL w `.env`, program automatycznie pobierze archiwum.

## Uruchomienie

```bash
# Pierwszy raz - z treningiem
python zad16.py

# Kolejne uruchomienia - bez treningu
python zad16.py --skip-training --model-id ft:gpt-4o-mini-2024-07-18:personal::xxxxx
```

Program:
- Sprawdza czy są pliki lokalne i używa ich
- Jeśli nie ma - pobiera z URL w `.env`
- Obsługuje błędną nazwę pliku `incorect.txt`
- Dodaje seed do fine-tuningu zgodnie z dokumentacją


# Zadanie 17 i sekret 6.

w `.env` polecam ustawić llama-3.3-70b-instruct dla `MODEL_NAME_LM` i `MODEL_NAME_ANY` aczkolwiek zalecam pobawić się modelami lokalnymi przy zadaniu 17. Sekret 6 jest realizowany za pomocą kodu w Python i nie wymaga żadnego modelu do uruchomienia.

Uruchom zadanie:

```bash
python zad17.py --engine openai
```

Uruchom sekret:

```bash
python sec6.py
```

Jeśli chcesz tylko flagę, uruchom za pomocą:

```bash
python agent.py
```

wybierz silnik i wpisz: `run_task 17` lub `run_secret 6`.

# Zadanie 18

Przeczytaj plik README_WEBHOOK_PL.md Zalecam model: qwen3-asteria-14b-128k dla LM Studio i Anything LLM.

# Zadanie 19

## Kluczowe funkcjonalności:

1. **Multi-engine support** - zgodnie z pozostałymi plikami `zad*.py`:
   - OpenAI, Claude, Gemini, LMStudio, Anything
   - Automatyczne wykrywanie silnika
   - Vision models do OCR zamiast Tesseract

2. **Pipeline z LangGraph**:
   - `download_pdf_node` - pobiera PDF z notatnikiem
   - `extract_content_node` - ekstraktuje tekst ze stron 1-18
   - `ocr_page19_node` - wykonuje OCR na stronie 19 (obraz)
   - `fetch_questions_node` - pobiera pytania z API
   - `answer_questions_node` - generuje odpowiedzi używając LLM
   - `send_answers_node` - wysyła i obsługuje hinty

3. **Inteligentne odpowiadanie**:
   - Używa pełnego kontekstu notatnika
   - Obsługuje hinty z centrali (iteracyjne poprawianie)
   - Formatuje daty jako YYYY-MM-DD
   - Krótkie, konkretne odpowiedzi

4. **Wymagana instalacja pakietów**:

```bash
pip install frontend
pip install PyMuPDF
pip install Pillow
```

## Uruchomienie:
```bash
python zad19.py --engine openai
# Zamiast openai wybierz inny silnik
```

Kod automatycznie:
- Pobierze PDF z notatnikiem
- Przetworzy strony 1-18 jako tekst
- Wykona OCR na stronie 19
- Odpowie na pytania używając LLM
- Obsłuży ewentualne hinty z centrali
- Wyśle poprawne odpowiedzi

---

# English version

AI Devs 3 Reloaded tasks in Python and LangGraph and other tools

* **GPU Drivers:** On Windows, install the NVIDIA driver compatible with WSL2 and CUDA (preferably the latest from the 525+ series). Make sure Windows detects the GPU, and WSL has access to it (the `nvidia-smi` command in WSL2 should show the RTX 3060).
* **Power Supply and eGPU:** A 650 W PSU is sufficient for an RTX 3060. Since there is no NVLink, we can't split computation across GPUs - the entire model must fit on one card (or partially into RAM). In practice, **gpu\_layers=99** (over 98%) puts most of the network on the GPU, requiring quantization and memory tuning to stay under the 12 GB VRAM limit.

# WSL2

Install Windows Subsystem for Linux via PowerShell:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
````

In Windows, enable the **WSL2 feature** (PowerShell: `wsl --install`) and add the **Ubuntu 22.04 / 24.04 or newer** distribution from the Microsoft Store. Yes, NVIDIA CUDA installation doesn't work on Debian 12 😂. Then in WSL:

```bash
sudo apt update
sudo apt install -y build-essential dkms cmake git python3 python3-pip nvidia-cuda-toolkit nvidia-cuda-dev libcurl4-openssl-dev curl jq unzip zipalign
```

In PowerShell execute

```powershell
wsl --shutdown
```

After rebooting, run the `nvidia-smi` command in WSL window; if you see a list of processors, the GPU is ready.

To run the tasks:

Rename `_env` to `.env`, modify it, add missing values and save it.

Executhe the below commands

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-pol ffmpeg
git clone https://github.com/sysadmin-info/reloaded.git
cd reloaded
mkdir venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 agent.py
```

Choose the remote or local provider and then type `run_task N` where `N` is a number of task , so 1, 2, 3 ... etc.

# PowerShell - Windows 11

If you prefer PowerShell then below you have a guide:

Install Python 3.11.9 from here: [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)

Execute the below commands:

```powershell
py -3.11 -m venv "dev-venv"
\dev-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install openai-whisper requests opencv-python pytesseract langdetect langchain-core langchain-openai langchain-google-genai langgraph dotenv bs4 google.generativeai
```

Install choco using this guide: [Chocolatey](https://docs.chocolatey.org/en-us/choco/setup/ )

and then install ffmpeg and tesseract

Run in PowerShell as administrator:

```powershell
choco install ffmpeg tesseract
```

### LM Studio and Anything LLM

Download and install:

[LM Studio](https://lmstudio.ai/)
[Anything LLM](https://anythingllm.com/)

First download some models in LM Studio, then learn how to use it. In Anything LLM set the backend to LM Studio from a dropdown list (it will automatically recognize available models).

Then run:

```powershell
cd reloaded
source venv/bin/activate
python3 agent.py
```

# How does agent.py work

What you're seeing (only the flag output when running via `agent.py`) is **intentional behavior** and the result of a carefully considered design.

### Why is the output "clean" (only the flag)?

1. **`agent.py`** runs the tasks (`zadN.py`) as a **subprocess** with redirected output (stdout), meaning it captures the entire script output and then:

   * Parses only the flag(s) from the output (`{{FLG:...}}`).
   * **All other messages and logs are ignored** (unless an error occurs).
   * Only the final message with the flag appears on screen (`🏁 Flag found: ... - finishing task.`), instead of the full verbose console from `zadN.py`.
   * See this snippet from the `_execute_task` and `run_task` functions in `agent.py`:

     ```python
     output_full = result.stdout.rstrip()
     flags = re.findall(r"\{\{FLG:.*?\}\}|FLG\{.*?\}", output_full)
     if flags:
         # ... prints only the flags
     ```
   * **Messages like "⚠️ Only OpenAI supports embeddings..." will appear only when you run `zad12.py` directly**, not via the agent.

2. **This means you're free to add as many print/debug statements to `zad12.py` as you like** (warnings, info, etc.)—the agent will only display the flag, keeping console output clean.

---

### **How does it work?**

* **Indirect execution via the agent**: shows only the flag.
* **Direct execution** (`python zadN.py`): shows all prints and logs.

---

### **What if you want to change this behavior?**

1. **Want to see more logs when using the agent?**
   – You can easily modify the `_execute_task` function in `agent.py` to also print `stdout` when no flags are found, or even always (just add a print line with the full output).

2. **Want to keep it as is?**
   – The current behavior is **perfect for CTF/AI-Devs competitions** – flag-only output, clean console, automatic flag registration.

---

**In summary:**
This is intentional, well-thought-out behavior.  
Want to see full verbose output? Run the script directly.  
Want clean flag-only output? Use the agent.  
Everything is configured **perfectly**!

---

If you'd like a “hybrid” version — e.g., additional info shown when running via agent — let me know and I’ll provide a ready-made change for `agent.py`.  
You can also use `print(..., file=sys.stderr)` and redirect it if you want debug info on stderr and flags on stdout (for your own automation setups).

# Task 12 - zad12.py

## Running Qdrant in Docker:

```bash
# Basic Qdrant launch (similar to your existing setup)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant

# With data persistence (recommended)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Restart an existing container
docker restart hardcore_jackson
````

## Updating `zad12.py` for Docker-based Qdrant:

Change the Qdrant initialization line from:

```python
qdrant_client = QdrantClient(":memory:")
```

to:

```python
qdrant_client = QdrantClient("localhost", port=6333)
```

## Docker Desktop in WSL2 with Ubuntu 24.04.2:

1. **Install Docker Desktop on Windows**:

   * Download from: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   * During installation, check the option "Use WSL 2 instead of Hyper-V"

2. **Enable integration with WSL2**:

   * Open Docker Desktop
   * Go to Settings → Resources → WSL Integration
   * Enable "Enable integration with my default WSL distro"
   * Select your Ubuntu distribution

3. **Check in WSL2**:

   ```bash
   # Should work without sudo
   docker version
   docker ps
   ```

4. **If you encounter permission issues**:

   ```bash
   # Add your user to the docker group
   sudo usermod -aG docker $USER

   # Log out and back in, or:
   newgrp docker
   ```

## Useful Qdrant commands:

```bash
# Check logs
docker logs hardcore_jackson

# Check collection status via API
curl http://localhost:6333/collections

# Should return something like:
# {"result":{"collections":[]},"status":"ok","time":0.000123}

# Qdrant dashboard (if port 6334 is open)
# Open in your browser: http://localhost:6333/dashboard
```

### Activate your virtual environment (if using)

```bash
source venv/bin/activate  # or your specific name
```

### Install qdrant-client

```bash
pip install qdrant-client
```

### In your `.env` file, add:

```bash
# Defaults to localhost:6333

QDRANT\_URL=localhost
QDRANT\_PORT=6333

# Or for in-memory (without Docker)

# QDRANT\_URL=\:memory:
```

## Configuration Options Summary:

### 1. **In-memory** (simplest, for testing)

```bash
# In .env
QDRANT_HOST=:memory:
```

* ✅ No Docker required
* ✅ Works instantly
* ❌ Data is lost after shutdown

### 2. **Local Docker** (WSL2, Linux, macOS)

```bash
# Run Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# In .env (or leave defaults)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

* ✅ Data persistence
* ✅ Full functionality
* ⚠️ Requires Docker

### 3. **Remote Docker** (VM, server)

```bash
# In .env
QDRANT_HOST=192.168.1.100  # IP of your server
QDRANT_PORT=6333
```

### 4. **Qdrant Cloud**

```bash
# In .env
QDRANT_URL=https://xyz.qdrant.io
QDRANT_API_KEY=your_qdrant_key
```

The code will automatically detect the configuration and display appropriate messages.
If the Docker connection fails, you’ll get helpful diagnostics.

# Task 14 - zad14.py and secret 4.

Install Docker, install container with neo4j

```bash
docker run -d   --name neo4j   -p 7474:7474 -p 7687:7687   -v $HOME/neo4j/data:/data   -v $HOME/neo4j/logs:/logs   -v $HOME/neo4j/import:/import   -v $HOME/neo4j/plugins:/plugins   -e NEO4J_AUTH=neo4j/YourStrongPassword123   neo4j:5
```

Install neo4j and YouTube downloader

```bash
pip install neo4j yt-dlp
```

# Task 15 and Secret 5.

Try running with:

```bash
python zad15.py --engine openai
```

If there are still problems with the tokens, you can use the small image version:

```bash
python zad15.py --engine openai --use-small
```

If you just want a flag, run with:

```bash
python agent.py
```

select the engine and type: `run_task 15`.

For LM Studio and Anything LLM download [qwen2.5-vl-7b](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

# Task 16

If you have a valid URL in `.env`, the program will automatically download the archive.

## Run

```bash
# First time - with training
python zad16.py

# Subsequent runs - without training
python zad16.py --skip-training --model-id ft:gpt-4o-mini-2024-07-18:personal::xxxxx
```

Program:
- Checks if there are local files and uses them
- If there are not - retrieves from the URL in `.env`.
- Handles incorrect file name `incorect.txt`.
- Adds seed to fine-tuning according to the documentation

# Task 17 and secret 6.

In `.env` I recommend setting llama-3.3-70b-instruct for `MODEL_NAME_LM` and `MODEL_NAME_ANY` although I recommend playing with local models at task 17. Secret 6 is implemented using code in Python and does not require any model to run.

Run the task with:

```bash
python zad17.py --engine openai
```

Run the secret with:

```bash
python sec6.py
```

If you just want a flag, run with:

```bash
python agent.py
```

select the engine and type: `run_task 17` or `run_secret 6`.

# Task 18

Read the README_WEBHOOK_EN.md file. I recommend model: qwen3-asteria-14b-128k dfor LM Studio and Anything LLM.

# Task 19

## Key functionalities:

1. **Multi-engine support** – in line with other `zad*.py` files:
   - OpenAI, Claude, Gemini, LMStudio, Anything
   - Automatic engine detection
   - Vision models for OCR instead of Tesseract

2. **Pipeline using LangGraph**:
   - `download_pdf_node` – downloads the notebook PDF
   - `extract_content_node` – extracts text from pages 1-18
   - `ocr_page19_node` – performs OCR on page 19 (image)
   - `fetch_questions_node` – fetches questions from the API
   - `answer_questions_node` – generates answers using LLM
   - `send_answers_node` – submits answers and handles hints

3. **Intelligent answering**:
   - Uses the full context of the notebook
   - Handles hints from the central system (iterative corrections)
   - Formats dates as YYYY-MM-DD
   - Short, concise answers

4. **Required package installation**:

```bash
pip install frontend
pip install PyMuPDF
pip install Pillow
````

## Execution:

```bash
python zad19.py --engine openai
# Replace "openai" with a different engine as needed
```

The code will automatically:

* Download the notebook PDF
* Process pages 1-18 as text
* Perform OCR on page 19
* Answer questions using the LLM
* Handle any hints from the central system
* Submit the correct answers