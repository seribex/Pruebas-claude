"""
Descarga Wikipedia en espanol COMPLETA (1.8 millones de articulos, ~3.5 GB
de texto limpio) para entrenar a Atlas Lumerak a escala real.

Esto reemplaza a fetch_wikipedia.py, que solo bajaba las introducciones de
articulos al azar (~4 millones de caracteres). Aca bajamos los articulos
enteros: unas 250 veces mas texto.

Los archivos son grandes, asi que se procesan de a uno y se borra cada
descarga apenas se termina de usar, para no acumular gigas en disco. El
archivo de salida NO se sube a git (es demasiado grande) -- cada quien lo
genera en su maquina corriendo este script.
"""

import argparse
import os
import urllib.request

import pandas as pd

from text_cleaning import limpiar_texto

BASE_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/"
    "refs%2Fconvert%2Fparquet/20231101.es/train/{:04d}.parquet"
)
N_ARCHIVOS = 13
MIN_LARGO_ARTICULO = 200


def main():
    parser = argparse.ArgumentParser(description="Prepara Wikipedia en espanol completa.")
    parser.add_argument("--out", default="atlas_lumerak/data/wikipedia_es_completa.txt")
    parser.add_argument("--tmp_dir", default="atlas_lumerak/data/_tmp_wikipedia")
    parser.add_argument("--max_archivos", type=int, default=N_ARCHIVOS,
                        help="Cuantos de los 13 archivos procesar (menos = corpus mas chico).")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total_chars = 0
    total_articulos = 0

    with open(args.out, "w", encoding="utf-8") as salida:
        for i in range(args.max_archivos):
            url = BASE_URL.format(i)
            ruta_tmp = os.path.join(args.tmp_dir, f"{i:04d}.parquet")

            print(f"[{i + 1}/{args.max_archivos}] Descargando {url.split('/')[-1]}...", flush=True)
            urllib.request.urlretrieve(url, ruta_tmp)

            df = pd.read_parquet(ruta_tmp, columns=["text"])
            for texto in df["text"]:
                if not texto or len(texto) < MIN_LARGO_ARTICULO:
                    continue
                limpio = limpiar_texto(texto.strip())
                salida.write(limpio + "\n\n")
                total_chars += len(limpio)
                total_articulos += 1

            # Borrar la descarga apenas se uso, para no acumular gigas.
            os.remove(ruta_tmp)
            print(f"    acumulado: {total_articulos:,} articulos | {total_chars:,} caracteres", flush=True)

    try:
        os.rmdir(args.tmp_dir)
    except OSError:
        pass

    print(f"\nListo. {total_articulos:,} articulos, {total_chars:,} caracteres en {args.out}")
    print(f"(~{total_chars / 5.5 / 1e6:.0f} millones de palabras aproximadamente)")


if __name__ == "__main__":
    main()
