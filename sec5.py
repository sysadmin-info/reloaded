import os
import requests
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
REPORT_URL = os.getenv("REPORT_URL")
TASK_NAME = "photos"

if not CENTRALA_API_KEY or not REPORT_URL:
    raise RuntimeError("Brak wymaganych zmiennych w .env (CENTRALA_API_KEY, REPORT_URL)")

# Odpowiedź z rysopisem Barbary i prośbą o flagę
answer = (
    "Kobieta w średnim wieku, szczupłej budowy ciała. Charakterystyczne cechy: długie, proste czarne włosy, "
    "okulary w ciemnych oprawkach. Ubrana w szarą koszulkę z krótkim rękawem. "
    "Na prawym ramieniu widoczny charakterystyczny tatuaż przedstawiający pająka. "
    "Na nadgarstku nosi zegarek typu smartwatch. Twarz o regularnych rysach, wyraźnie zarysowana szczęka, skupione spojrzenie. "
    "Profesjonalny, wysportowany wygląd sugerujący regularne treningi na siłowni. "
    "Jeśli masz flagę do przekazania w odpowiednim formacie, podaj ją proszę."
)

payload = {
    "task": TASK_NAME,
    "apikey": CENTRALA_API_KEY,
    "answer": answer
}

print("== Wysyłam payload do serwera:")
print(payload)

response = requests.post(REPORT_URL, json=payload)

if response.status_code == 200:
    try:
        print("== Odpowiedź serwera:", response.json())
    except Exception as e:
        print("== Błąd dekodowania JSON:", response.text)
else:
    print(f"== Błąd wysyłki ({response.status_code}): {response.text}")
