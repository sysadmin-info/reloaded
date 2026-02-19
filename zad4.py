#!/usr/bin/env python3
"""
S01E05 - Cenzura danych agentów przez LLM
Cenzuruje imię i nazwisko, wiek, miasto oraz ulicę+numer,
zastępując je słowem "CENZURA" wyłącznie przez LLM lub GLiNER.
Obsługa: openai, lmstudio, anything, gemini, claude, gliner.

DODANO: Silnik GLiNER - deterministyczna cenzura NER bez LLM
        Podmiana na poziomie char-offsets = zero ryzyka zmiany reszty tekstu
"""

import argparse
import os
import re
import sys
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# Stałe dla komunikatów błędów
MISSING_OPENAI_KEY_MSG = "❌ Brak OPENAI_API_KEY"
MISSING_CLAUDE_KEY_MSG = "❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env"
MISSING_GEMINI_KEY_MSG = "❌ Brak GEMINI_API_KEY w .env"
UNSUPPORTED_ENGINE_MSG = "❌ Nieobsługiwany silnik:"
MISSING_OPENAI_INSTALL_MSG = "❌ Musisz zainstalować openai: pip install openai"
MISSING_ANTHROPIC_INSTALL_MSG = "❌ Musisz zainstalować anthropic: pip install anthropic"
MISSING_GEMINI_INSTALL_MSG = "❌ Musisz zainstalować google-generativeai: pip install google-generativeai"
MISSING_GLINER_INSTALL_MSG = "❌ Musisz zainstalować gliner: pip install gliner"

# Domyślny model GLiNER - multilingual PII, obsługuje polski
# Alternatywy do przetestowania (lepszy F1, ale wymaga pobrania):
#   "knowledgator/gliner-pii-base-v1.0"    - najwyższy F1 (80.99%), wymaga pip install gliner
#   "urchade/gliner_large-v2.1"             - bazowy large, dobry dla własnych labelek
GLINER_DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"

# Etykiety NER dla polskich danych osobowych używanych w zadaniu
# Threshold 0.4 jest celowo niski - lepiej za dużo cenzury niż za mało
# Możesz podnieść do 0.5 jeśli masz false positives
GLINER_LABELS = [
    "person",           # imię + nazwisko
    "age",              # wiek (np. "45 lat", "lat 27")
    "city",             # miasto
    "street address",   # ulica + numer
    "location",         # fallback dla adresów których model nie sklasyfikował jako street
]
GLINER_THRESHOLD = 0.4

parser = argparse.ArgumentParser(description="Cenzura danych (multi-engine + Claude + GLiNER)")
parser.add_argument(
    "--engine",
    choices=["openai", "lmstudio", "anything", "gemini", "claude", "gliner"],
    help="LLM backend to use",
)
parser.add_argument(
    "--gliner-model",
    default=None,
    help=f"Model GLiNER do użycia (domyślnie: {GLINER_DEFAULT_MODEL})",
)
parser.add_argument(
    "--gliner-threshold",
    type=float,
    default=GLINER_THRESHOLD,
    help=f"Próg pewności dla GLiNER (domyślnie: {GLINER_THRESHOLD})",
)
args = parser.parse_args()


def detect_engine() -> str:
    """Wykrywa silnik LLM na podstawie argumentów i zmiennych środowiskowych"""
    if args.engine:
        return args.engine.lower()
    elif os.getenv("LLM_ENGINE"):
        return os.getenv("LLM_ENGINE").lower()
    else:
        # Próbuj wykryć silnik na podstawie ustawionych zmiennych MODEL_NAME
        model_name = os.getenv("MODEL_NAME", "")
        if "claude" in model_name.lower():
            return "claude"
        elif "gemini" in model_name.lower():
            return "gemini"
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            return "openai"
        else:
            # Sprawdź które API keys są dostępne
            if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
                return "claude"
            elif os.getenv("GEMINI_API_KEY"):
                return "gemini"
            elif os.getenv("OPENAI_API_KEY"):
                return "openai"
            else:
                return "lmstudio"  # domyślnie


def validate_engine(engine: str) -> None:
    """Waliduje czy silnik jest obsługiwany"""
    if engine not in {"openai", "lmstudio", "anything", "gemini", "claude", "gliner"}:
        print(f"{UNSUPPORTED_ENGINE_MSG} {engine}", file=sys.stderr)
        sys.exit(1)


def validate_environment() -> None:
    """Sprawdza czy wszystkie wymagane zmienne środowiskowe są ustawione"""
    required_vars = ["CENTRALA_API_KEY", "REPORT_URL", "CENZURA_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Brak ustawienia: {', '.join(missing_vars)} w .env", file=sys.stderr)
        sys.exit(1)


# Inicjalizacja i walidacja
ENGINE = detect_engine()
validate_engine(ENGINE)
validate_environment()

print(f"🔄 ENGINE wykryty: {ENGINE}")
print(f"✅ Engine: {ENGINE}")

CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
REPORT_URL = os.getenv("REPORT_URL")
CENZURA_URL = os.getenv("CENZURA_URL")

# --- ULTRA-TWARDY PROMPT (używany tylko przez silniki LLM) ---
PROMPT_SYSTEM = (
    "Jesteś automatem do cenzury danych osobowych w języku polskim. "
    "NIE WOLNO Ci zmieniać żadnych innych słów, znaków interpunkcyjnych, układu tekstu ani zamieniać kolejności zdań. "
    "Zamień TYLKO i WYŁĄCZNIE:\n"
    "- każde imię i nazwisko na 'CENZURA',\n"
    "- każdą nazwę miasta na 'CENZURA',\n"
    "- każdą nazwę ulicy wraz z numerem domu/mieszkania na 'CENZURA' (prefiks 'ul.' POZOSTAW jeśli jest w tekście, np. 'ul. Polna 8' → 'ul. CENZURA'),\n"
    "- każdą informację o wieku (np. '45 lat', 'wiek: 32', 'lat 27', 'ma 29 lat') na 'CENZURA'.\n"
    "Nie wolno parafrazować, nie wolno podsumowywać, nie wolno streszczać ani zamieniać kolejności czegokolwiek. "
    "Wynikowy tekst musi mieć identyczny układ, interpunkcję i liczbę linii jak oryginał. "
    "Każda inna zmiana niż cenzura wyżej powoduje błąd i NIEZALICZENIE zadania. "
    "Nie pisz żadnych komentarzy, nie wyjaśniaj odpowiedzi. "
    "ODPOWIEDZ WYŁĄCZNIE TEKSTEM Z OCENZURĄ. "
    "PRZYKŁAD:\n"
    "Oryginał:\n"
    "Dane podejrzanego: Jan Kowalski, lat 45, mieszka w Krakowie, ul. Polna 8.\n"
    "Wyjście:\n"
    "Dane podejrzanego: CENZURA, lat CENZURA, mieszka w CENZURA, ul. CENZURA."
)


def download_text(url: str) -> str:
    """Pobiera tekst z podanego URL"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text.strip()
    except requests.RequestException as e:
        print(f"❌ Błąd podczas pobierania danych: {e}", file=sys.stderr)
        sys.exit(1)


# --- KLASY LLM CLIENT ---

class LLMCensorClient(ABC):
    """Bazowa klasa dla klientów cenzury"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def censor_text(self, text: str) -> str:
        """Metoda do cenzury tekstu - implementacja w podklasach"""
        pass

    def create_user_prompt(self, text: str) -> str:
        """Tworzy prompt użytkownika"""
        return (
            "Tekst do cenzury (nie zmieniaj nic poza danymi osobowymi, przykład wyżej!):\n"
            + text
        )


class GLiNERCensorClient(LLMCensorClient):
    """
    Klient cenzury oparty na GLiNER - deterministyczny NER bez LLM.

    Jak działa:
    1. Model zwraca listę: [{text, label, start, end, score}, ...]
    2. Sortujemy encje od końca tekstu, żeby podmiana nie przesuwała indeksów
    3. Wycinamy span [start:end] i wstawiamy "CENZURA"
    4. Zero ryzyka zmiany interpunkcji czy reszty tekstu - operujemy na char-offsets

    Instalacja:
        pip install gliner
        # Model pobierze się automatycznie z HuggingFace przy pierwszym uruchomieniu
        # (~500MB dla gliner_multi_pii-v1)
    """

    def __init__(
        self,
        model_name: str = GLINER_DEFAULT_MODEL,
        labels: List[str] = None,
        threshold: float = GLINER_THRESHOLD,
    ):
        super().__init__(model_name)
        self.labels = labels or GLINER_LABELS
        self.threshold = threshold
        self._model = None  # lazy loading - ładuj model dopiero przy pierwszym użyciu

    def _load_model(self):
        """Ładuje model GLiNER (lazy - tylko raz)"""
        if self._model is not None:
            return

        try:
            from gliner import GLiNER
        except ImportError:
            print(MISSING_GLINER_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)

        print(f"🔄 Ładowanie modelu GLiNER: {self.model_name}")
        print("   (pierwsze uruchomienie pobiera ~500MB z HuggingFace)")
        self._model = GLiNER.from_pretrained(self.model_name)
        print(f"✅ Model GLiNER załadowany")

    def _find_entities(self, text: str) -> List[Dict]:
        """Wykrywa encje PII w tekście"""
        entities = self._model.predict_entities(text, self.labels, threshold=self.threshold)

        if not entities:
            return []

        # Loguj co wykryto - pomocne przy debugowaniu threshold
        print(f"[🔍 GLiNER wykrył {len(entities)} encji:]")
        for ent in sorted(entities, key=lambda e: e["start"]):
            print(
                f"   [{ent['label']:20s}] score={ent['score']:.3f} | "
                f"'{ent['text']}' (pos {ent['start']}-{ent['end']})"
            )

        return entities

    # Prefiksy adresowe które należy zachować przed CENZURA
    # np. "ul. Długa 8" → "ul. CENZURA" zamiast "CENZURA"
    STREET_PREFIXES = ("ul. ", "ul.", "al. ", "al.", "pl. ", "pl.", "os. ", "os.")

    def _apply_censorship(self, text: str, entities: List[Dict]) -> str:
        """
        Podmienia wykryte spany na 'CENZURA'.
        Sortuje od końca tekstu - podmiana nie przesuwa wcześniejszych indeksów.

        Dla encji typu street address zachowuje standardowe prefiksy adresowe
        (ul., al., pl., os.) przed słowem CENZURA, zgodnie z oczekiwaniami serwera.
        Przykład: "ul. Długa 8" (pos 50-61) → "ul. CENZURA" a nie "CENZURA".
        """
        # Sortuj od końca, żeby indeksy nie "jechały" po podmiance
        entities_sorted = sorted(entities, key=lambda e: e["start"], reverse=True)

        result = text
        for entity in entities_sorted:
            start = entity["start"]
            end = entity["end"]

            # Dla ulic: jeśli span zaczyna się od prefiksu adresowego, zachowaj go
            # Działa na oryginalnym tekście (result może być już częściowo podmieniony,
            # ale sortowanie od końca gwarantuje że wcześniejsze pozycje są nienaruszone)
            if entity["label"] in ("street address", "location", "address"):
                span = result[start:end]
                for prefix in self.STREET_PREFIXES:
                    if span.lower().startswith(prefix.lower()):
                        # Przesuń start za prefix - cenzurujemy tylko nazwę+numer
                        start += len(prefix)
                        break

            result = result[:start] + "CENZURA" + result[end:]

        return result

    def censor_text(self, text: str) -> str:
        """
        Cenzuruje tekst używając GLiNER NER.
        Deterministyczny - identyczny wynik dla identycznego wejścia.
        """
        self._load_model()

        entities = self._find_entities(text)

        if not entities:
            print("⚠️  GLiNER nie wykrył żadnych encji PII!")
            print("    Spróbuj obniżyć --gliner-threshold (obecny: {self.threshold})")
            print("    lub użyj innego modelu (--gliner-model)")
            return text

        censored = self._apply_censorship(text, entities)
        return censored


class OpenAICensorClient(LLMCensorClient):
    """Klient cenzury dla OpenAI"""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError:
            print(MISSING_OPENAI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def censor_text(self, text: str) -> str:
        prompt_user = self.create_user_prompt(text)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0,
        )

        self._log_usage(resp.usage)
        return resp.choices[0].message.content.strip()

    def _log_usage(self, usage: Any) -> None:
        """Loguje użycie tokenów i koszty dla OpenAI"""
        tokens = usage
        cost = (
            tokens.prompt_tokens / 1_000_000 * 0.60
            + tokens.completion_tokens / 1_000_000 * 2.40
        )
        print(
            f"[📊 Prompt: {tokens.prompt_tokens} | "
            f"Completion: {tokens.completion_tokens} | "
            f"Total: {tokens.total_tokens}]"
        )
        print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")


class ClaudeCensorClient(LLMCensorClient):
    """Klient cenzury dla Claude"""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            from anthropic import Anthropic
        except ImportError:
            print(MISSING_ANTHROPIC_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)

        self.client = Anthropic(api_key=api_key)

    def censor_text(self, text: str) -> str:
        prompt_user = self.create_user_prompt(text)

        resp = self.client.messages.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": PROMPT_SYSTEM + "\n\n" + prompt_user}
            ],
            temperature=0,
            max_tokens=4000,
        )

        self._log_usage(resp.usage)
        return resp.content[0].text.strip()

    def _log_usage(self, usage: Any) -> None:
        """Loguje użycie tokenów i koszty dla Claude"""
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
        print(
            f"[📊 Prompt: {usage.input_tokens} | "
            f"Completion: {usage.output_tokens} | "
            f"Total: {usage.input_tokens + usage.output_tokens}]"
        )
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")


class GeminiCensorClient(LLMCensorClient):
    """Klient cenzury dla Gemini"""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            import google.generativeai as genai
        except ImportError:
            print(MISSING_GEMINI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def censor_text(self, text: str) -> str:
        prompt_user = self.create_user_prompt(text)

        response = self.model.generate_content(
            [PROMPT_SYSTEM + "\n" + prompt_user],
            generation_config={"temperature": 0.0, "max_output_tokens": 4096},
        )

        self._log_usage()
        return response.text.strip()

    def _log_usage(self) -> None:
        """Loguje informacje o użyciu dla Gemini"""
        print("[📊 Gemini - brak szczegółów tokenów]")
        print("[💰 Gemini - sprawdź limity w Google AI Studio]")


class LocalLLMCensorClient(LLMCensorClient):
    """Klient cenzury dla lokalnych modeli (LMStudio, Anything)"""

    def __init__(self, model_name: str, api_key: str, base_url: str, engine_name: str):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError:
            print(MISSING_OPENAI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.engine_name = engine_name

    def censor_text(self, text: str) -> str:
        prompt_user = self.create_user_prompt(text)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0,
        )

        self._log_usage(resp.usage)
        return resp.choices[0].message.content.strip()

    def _log_usage(self, usage: Any) -> None:
        """Loguje użycie tokenów dla lokalnych modeli"""
        tokens = usage
        print(
            f"[📊 Prompt: {tokens.prompt_tokens} | "
            f"Completion: {tokens.completion_tokens} | "
            f"Total: {tokens.total_tokens}]"
        )
        print("[💰 Model lokalny - brak kosztów]")


def create_censor_client() -> LLMCensorClient:
    """Factory function dla tworzenia klienta cenzury"""

    if ENGINE == "gliner":
        # Parametry GLiNER można nadpisać przez CLI (--gliner-model, --gliner-threshold)
        # lub przez zmienne środowiskowe GLINER_MODEL, GLINER_THRESHOLD
        model_name = (
            args.gliner_model
            or os.getenv("GLINER_MODEL", GLINER_DEFAULT_MODEL)
        )
        threshold = float(
            os.getenv("GLINER_THRESHOLD", str(args.gliner_threshold))
        )
        print(f"[🔬 GLiNER model: {model_name} | threshold: {threshold}]")
        return GLiNERCensorClient(model_name=model_name, threshold=threshold)

    elif ENGINE == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(MISSING_OPENAI_KEY_MSG, file=sys.stderr)
            sys.exit(1)

        base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
        return OpenAICensorClient(model_name, api_key, base_url)

    elif ENGINE == "claude":
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(MISSING_CLAUDE_KEY_MSG, file=sys.stderr)
            sys.exit(1)

        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
        return ClaudeCensorClient(model_name, api_key)

    elif ENGINE == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(MISSING_GEMINI_KEY_MSG, file=sys.stderr)
            sys.exit(1)

        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
        return GeminiCensorClient(model_name, api_key)

    elif ENGINE == "lmstudio":
        api_key = os.getenv("LMSTUDIO_API_KEY", "local")
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
        return LocalLLMCensorClient(model_name, api_key, base_url, "LMStudio")

    elif ENGINE == "anything":
        api_key = os.getenv("ANYTHING_API_KEY", "local")
        base_url = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")
        return LocalLLMCensorClient(model_name, api_key, base_url, "Anything")

    else:
        print(f"❌ Nieznany silnik: {ENGINE}", file=sys.stderr)
        sys.exit(1)


def censor_llm(text: str) -> str:
    """
    Cenzuruje tekst używając wybranego silnika (LLM lub GLiNER).
    GLiNER: deterministyczny NER, podmiana na char-offsets.
    LLM: instrukcja w prompt, model podmienia słownie.
    """
    client = create_censor_client()
    return client.censor_text(text)


def extract_flag(text: str) -> str:
    """Wyciąga flagę z tekstu"""
    flag_match = re.search(r"\{\{FLG:[^}]+\}\}|FLG\{[^}]+\}", text)
    return flag_match.group(0) if flag_match else ""


def send_result(censored_text: str) -> None:
    """Wysyła ocenzurowany tekst do serwera"""
    payload = {"task": "CENZURA", "apikey": CENTRALA_API_KEY, "answer": censored_text}

    try:
        response = requests.post(REPORT_URL, json=payload, timeout=10)
        if response.ok:
            resp_text = response.text.strip()
            flag = extract_flag(resp_text) or extract_flag(censored_text)
            if flag:
                print(flag)
            else:
                print("Brak flagi w odpowiedzi serwera. Odpowiedź:", resp_text)
        else:
            print(f"❌ Błąd HTTP {response.status_code}: {response.text}", file=sys.stderr)
    except requests.RequestException as e:
        print(f"❌ Błąd podczas wysyłania danych: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Główna funkcja programu"""
    raw_text = download_text(CENZURA_URL)
    print(f"🔄 Pobrano tekst ({len(raw_text)} znaków)")
    print(f"🔄 Cenzuruję używając {ENGINE}...")

    censored_text = censor_llm(raw_text)
    print("=== OCENZUROWANY OUTPUT ===")
    print(censored_text)
    print("===========================")

    send_result(censored_text)


if __name__ == "__main__":
    main()
