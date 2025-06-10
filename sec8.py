#!/usr/bin/env python3
"""
Rozwiązuje zadanie GPS – zamienia współrzędne z pliku na flagę.
"""
import os
import requests
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)
    gps_url = os.getenv("REPORT_URL").replace('/report', '/gps')

    # 1. Pobierz zagadkę dla userID 443
    resp = requests.post(gps_url, json={"userID": 443})
    data = resp.json()
    url = data.get("message", {}).get("lon")
    if not url:
        print("❌ Nie udało się pobrać URL z zagadką.")
        return

    # 2. Pobierz współrzędne
    txt = requests.get(url).text
    lines = [line.strip() for line in txt.splitlines()]
    codes = []
    for line in lines:
        try:
            # bierzemy część całkowitą, jeśli to liczba
            codes.append(int(float(line)))
        except ValueError:
            continue

    # 3. Zamiana kodów ASCII na flagę
    flag = "".join(chr(c) for c in codes)

    # 4. Wypisz flagę (wyłapywaną przez agent.py)
    print(flag)

if __name__ == "__main__":
    main()
