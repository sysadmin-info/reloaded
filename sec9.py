#!/usr/bin/env python3
import os
import base64
import re
from dotenv import load_dotenv
import dns.resolver

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv(override=True)

DNS_URL = os.getenv("DNS_URL")
if not DNS_URL:
    raise RuntimeError("Brak zmiennej DNS_URL w pliku .env")

def hex_to_ascii(hexstr):
    chars = hexstr.split(':')
    return ''.join([chr(int(c, 16)) for c in chars if len(c)==2])

def main():
    records = dns.resolver.resolve(DNS_URL, 'TXT')
    decoded_lines = []

    for r in records:
        b64str = b''.join(r.strings).decode('utf-8').strip()
        try:
            txt = base64.b64decode(b64str).decode('utf-8', errors='replace')
            decoded_lines.append(txt)
        except Exception:
            continue

    all_text = '\n'.join(decoded_lines)
    print("=== Dekodowane linie ===")
    print(all_text)
    print("========================")

    # Znajdź linię z hex
    m = re.search(r'LG:(([0-9A-Fa-f]{2}:?)+)}', all_text)
    flag_ascii = None
    if m:
        flag_ascii = hex_to_ascii(m.group(1))
    # Znajdź początek flagi (np. "Flaga to: {{F" lub "FLG:")
    m2 = re.search(r'Flaga to: *\{\{?F', all_text)
    if flag_ascii and m2:
        flag = f"{{{{FLG:{flag_ascii}}}}}"
        print(flag)
    else:
        print("❌ Nie znaleziono flagi!")

if __name__ == "__main__":
    main()
