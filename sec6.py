#!/usr/bin/env python3
"""
SEC6 - Znalezienie flagi z portfolio
"""
import os
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

SOFTO_URL = os.getenv("SOFTO_URL")

# Generator URLi portfolio_1..portfolio_6
portfolio_urls = [
    f"{SOFTO_URL}/portfolio_{n}_{hashlib.md5(str(n).encode()).hexdigest()}"
    for n in range(1, 7)
]

def extract_flag(text):
    """
    Zwraca FLG w formacie {{FLG:...}} lub FLG{...} albo None
    """
    import re
    match = re.search(r'(\{\{FLG:[^}]+\}\}|FLG\{[^}]+\})', text)
    return match.group(1) if match else None

def main():
    for url in portfolio_urls:
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            flag = extract_flag(r.text)
            if flag:
                print(flag)
                return
        except Exception:
            continue
    print("Brak flagi.")

if __name__ == "__main__":
    main()
