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

Następnie uruchom:

```powershell
cd reloaded
source venv/bin/activate
python3 agent.py
```

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

Then run:

```powershell
cd reloaded
source venv/bin/activate
python3 agent.py
```
