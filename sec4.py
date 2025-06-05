#!/usr/bin/env python3
"""
Sekretne zadanie z odwróconym audio Vimeo — Wersja uniwersalna!
Obsługuje: openai, lmstudio, anything, gemini, claude
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import yt_dlp

load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Vimeo Audio Processor - sekretne zadanie")
parser.add_argument("--engine", choices=["openai", "lmstudio", "anything", "gemini", "claude"], help="LLM backend to use")
args = parser.parse_args()

ENGINE = None
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

VIMEO_URL = os.getenv("VIMEO_URL")
if not VIMEO_URL:
    print("❌ Brak VIMEO_URL w .env", file=sys.stderr)
    sys.exit(1)

class VimeoExtractor:
    def __init__(self, download_path="./downloads"):
        self.download_path = Path(download_path)
        self.download_path.mkdir(exist_ok=True)

    def download_audio_only(self, url):
        opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.download_path / '%(title)s_audio.%(ext)s'),
            'writeinfojson': True,
            'writethumbnail': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
                print(f"✅ Audio extracted to {self.download_path}")
                audio_files = list(self.download_path.glob("*_audio.mp3"))
                if audio_files:
                    return audio_files[-1]
                else:
                    print("❌ No audio file found after download")
                    return None
        except Exception as e:
            print(f"❌ Error downloading audio: {e}")
            return None

class AudioProcessor:
    def __init__(self, engine="openai"):
        self.engine = engine.lower()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="audio_proc_"))
        self.setup_llm_client()

    def setup_llm_client(self):
        if self.engine == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
            self.model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
            if not api_key:
                raise ValueError("❌ Missing OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url=api_url)
        elif self.engine == "lmstudio":
            from openai import OpenAI
            api_key = os.getenv("LMSTUDIO_API_KEY", "local")
            api_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
            self.model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
            self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=120)
        elif self.engine == "anything":
            from openai import OpenAI
            api_key = os.getenv("ANYTHING_API_KEY", "local")
            api_url = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
            self.model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")
            self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=120)
        elif self.engine == "claude":
            from anthropic import Anthropic
            api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("❌ Missing CLAUDE_API_KEY or ANTHROPIC_API_KEY")
            self.model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
            self.claude_client = Anthropic(api_key=api_key)
        elif self.engine == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("❌ Missing GEMINI_API_KEY")
            self.model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
            genai.configure(api_key=api_key)
            self.model_gemini = genai.GenerativeModel(self.model_name)
        else:
            raise ValueError(f"❌ Unsupported engine: {self.engine}")
        print(f"✅ Initialized {self.engine} with model: {self.model_name}")

    def extract_audio_segment(self, audio_path: Path, start_time: float, end_time: float = None) -> Path:
        output_path = self.temp_dir / f"segment_{start_time}s.mp3"
        cmd = ["ffmpeg", "-i", str(audio_path), "-ss", str(start_time)]
        if end_time:
            cmd.extend(["-t", str(end_time - start_time)])
        cmd.extend([
            "-acodec", "mp3", "-ar", "16000", "-ac", "1",
            "-y", str(output_path)
        ])
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✅ Extracted segment: {start_time}s to {end_time or 'end'}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Error extracting segment: {e}")
            return None

    def reverse_audio(self, audio_path: Path) -> Path:
        output_path = self.temp_dir / f"reversed_{audio_path.name}"
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-af", "areverse",
            "-y", str(output_path)
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✅ Reversed audio: {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Error reversing audio: {e}")
            return None

    def transcribe_audio(self, audio_path: Path, language="fr") -> str:
        print(f"🎤 Transcribing audio ({language})...")
        # OpenAI, LM Studio, Anything - próbujemy transkrypcji
        if self.engine in ["lmstudio", "anything"]:
            print("⚠️ LM Studio / Anything LLM NIE obsługuje transkrypcji audio (Whisper API).")
            print("🔄 Przełączam na lokalny openai-whisper...")
            try:
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(str(audio_path), language=language)
                text = result["text"].strip()
                print(f"✅ Lokalna transkrypcja whisper: {text}")
                return text
            except Exception as e:
                print(f"❌ Błąd lokalnej transkrypcji whisper: {e}")
                return ""
        elif self.engine in ["openai"]:
            try:
                transcribe_client = self.client
                with open(audio_path, 'rb') as f:
                    response = transcribe_client.audio.transcriptions.create(
                        file=f,
                        model="whisper-1",
                        response_format="text",
                        language=language
                    )
                    text = getattr(response, 'text', response)
                    print(f"✅ Transcription completed: {len(text)} characters")
                    return text.strip()
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                return ""
        elif self.engine in ["claude", "gemini"]:
            print(f"❌ Whisper transcription not available for {self.engine} engine")
            # Fallback na openai whisper jeśli mamy klucz
            if os.getenv("OPENAI_API_KEY"):
                print("🔄 Fallback: Using OpenAI Whisper for transcription...")
                try:
                    from openai import OpenAI
                    fallback_client = OpenAI(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
                    )
                    with open(audio_path, 'rb') as f:
                        response = fallback_client.audio.transcriptions.create(
                            file=f,
                            model="whisper-1",
                            response_format="text",
                            language=language
                        )
                    text = getattr(response, 'text', response).strip()
                    print(f"✅ Fallback transcription completed: {len(text)} characters")
                    return text
                except Exception as e:
                    print(f"❌ Fallback transcription failed: {e}")
                    return ""
            else:
                print("❌ Brak transkrypcji — brak klucza OpenAI.")
                return ""
        else:
            print(f"❌ Unknown engine: {self.engine}")
            return ""

    def translate_and_analyze(self, french_text: str, question: str = None) -> str:
        if not french_text:
            return "❌ No text to analyze"
        system_prompt = (
            "You will receive a French sentence. "
            "Respond ONLY with the most essential English word or phrase that describes what the sentence refers to. "
            "If it's about the blue thing above your head, respond with 'sky'. "
            "If it's about a flag, respond with 'flag'. "
            "If possible, respond with just ONE word (in English)."
        )
        user_prompt = french_text.strip()
        print(f"🤖 Analyzing with {self.engine}...")
        print(f"LLM USER PROMPT: {user_prompt!r}")
        try:
            if self.engine in ["openai", "lmstudio", "anything"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            elif self.engine == "claude":
                response = self.claude_client.messages.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                return response.content[0].text.strip()
            elif self.engine == "gemini":
                response = self.model_gemini.generate_content(
                    [system_prompt, user_prompt],
                    generation_config={"temperature": 0.1, "max_output_tokens": 512}
                )
                return response.text.strip()
        except Exception as e:
            print(f"❌ LLM analysis error: {e}")
            return f"Error during analysis: {e}"

    def cleanup(self):
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

class IntegratedProcessor:
    def __init__(self, engine="openai", download_dir="./vimeo_downloads"):
        self.vimeo_extractor = VimeoExtractor(download_dir)
        self.audio_processor = AudioProcessor(engine)
    def process_vimeo_url(self, start_time: float = 58, end_time: float = 61,
                         question: str = None, skip_download: bool = False) -> str:
        print("🚀 Starting Vimeo Audio Processing Pipeline")
        print("=" * 50)
        url = VIMEO_URL
        print(f"📺 Processing URL from .env: {url}")
        print("1/4 📥 Downloading audio from Vimeo...")
        audio_path = self.vimeo_extractor.download_audio_only(url)
        if not audio_path:
            return "❌ Failed to download audio from Vimeo"
        print("2/4 🔄 Processing reversed French segment...")
        segment_path = self.audio_processor.extract_audio_segment(audio_path, start_time, end_time)
        reversed_path = self.audio_processor.reverse_audio(segment_path)
        print("3/4 🎤 Transcribing audio (fr)...")
        transcription = self.audio_processor.transcribe_audio(reversed_path, language="fr")
        print(f"🇫🇷 French transcription: {transcription}")
        if not transcription or transcription.strip().startswith("{\"error\"") or transcription.strip().lower() in {"none", ""}:
            print("❌ Brak poprawnej transkrypcji! Przerywam pipeline.")
            return "❌ Failed to transcribe audio"
        fraza = transcription.strip()
        analysis = self.audio_processor.translate_and_analyze(fraza)
        print("3/4 🔍 Extracting key information...")
        flag_content = self.extract_flag_content(analysis)
        translations = {'sky': 'niebo'}
        polska_flaga = translations.get(flag_content.lower(), flag_content)
        print("4/4 🎯 Formatting final result...")
        formatted_result = f"{{{{FLG:{polska_flaga.upper()}}}}}"
        return formatted_result

    def extract_flag_content(self, analysis_text: str) -> str:
        text = analysis_text.strip()
        text = text.replace('"', '').replace("'", "").replace('.', '').strip()
        if len(text) <= 20 and ' ' not in text:
            return text
        lines = text.split('\n')
        for line in lines:
            line = line.strip().lower()
            if not line:
                continue
            if line.startswith(('1.', '2.', '3.', 'translation:', 'answer:', 'key:', 'response:')):
                continue
            words = line.split()
            for word in words:
                word = word.strip('.,!?"\'').lower()
                if word in ['sky', 'niebo', 'ciel', 'himmel', 'blue', 'niebieski']:
                    return word
                if 3 <= len(word) <= 15:
                    return word
        words = text.split()
        for word in words:
            word = word.strip('.,!?\"\'()[]{}')
            if len(word) >= 3 and word.lower() not in ['the', 'and', 'but', 'for', 'with', 'this', 'that']:
                return word
        return text[:15].strip()
    def cleanup(self):
        self.audio_processor.cleanup()

def main():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found. Install with: sudo apt install ffmpeg")
        sys.exit(1)
    try:
        processor = IntegratedProcessor(ENGINE, "./vimeo_downloads")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    try:
        result = processor.process_vimeo_url(
            start_time=58,
            end_time=61,
            question="What is the flag or secret message?",
            skip_download=False
        )
        print(f"\n🎯 FINAL RESULT:")
        print("=" * 50)
        print(result)
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
