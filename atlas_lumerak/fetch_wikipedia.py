"""
Descarga un corpus de texto moderno en espanol: introducciones de
articulos aleatorios de Wikipedia en espanol, via su API publica.

Pedimos solo la introduccion de cada articulo (no el articulo completo)
porque la API de Wikipedia limita a 1 articulo completo por solicitud,
pero permite hasta 20 introducciones por solicitud -- mucho mas rapido
para juntar miles de articulos variados sin saturar sus servidores.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request

API_URL = "https://es.wikipedia.org/w/api.php"
USER_AGENT = "AtlasLumerakProject/1.0 (proyecto educativo de aprendizaje de IA, sin fines comerciales)"
MIN_EXTRACT_LEN = 200


def fetch_batch(batch_size: int = 20):
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(batch_size),
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exlimit": "max",
        "format": "json",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    return data.get("query", {}).get("pages", {}).values()


def main():
    parser = argparse.ArgumentParser(description="Descarga un corpus de Wikipedia en espanol.")
    parser.add_argument("--out", default="atlas_lumerak/data/wikipedia_es.txt")
    parser.add_argument("--target_chars", type=int, default=4_000_000)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    seen_ids = set()
    total_chars = 0
    articles_kept = 0
    requests_made = 0

    with open(args.out, "w", encoding="utf-8") as out_file:
        while total_chars < args.target_chars:
            try:
                pages = fetch_batch(20)
            except Exception as e:
                print(f"Aviso: fallo una solicitud ({e}), reintentando...")
                time.sleep(1.0)
                continue

            requests_made += 1
            for page in pages:
                page_id = page.get("pageid")
                extract = page.get("extract", "").strip()

                if page_id in seen_ids or len(extract) < MIN_EXTRACT_LEN:
                    continue
                if extract.lower().startswith("puede referirse a") or "puede referirse a" in extract[:120].lower():
                    continue

                seen_ids.add(page_id)
                out_file.write(extract + "\n\n")
                total_chars += len(extract)
                articles_kept += 1

            if requests_made % 20 == 0:
                print(f"solicitudes: {requests_made} | articulos guardados: {articles_kept} | caracteres: {total_chars:,}")

            time.sleep(args.delay)

    print(f"\nListo. {articles_kept} articulos, {total_chars:,} caracteres guardados en {args.out}")


if __name__ == "__main__":
    main()
