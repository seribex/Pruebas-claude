"""
PASO 12 (datos): descargar texto web en espanol de FineWeb-2.

Por que hace falta, si ya tenemos Wikipedia entera:

Atlas escribe como una enciclopedia porque Wikipedia es lo unico que ha
leido en su vida. No es una sospecha: en plena conversacion le salieron
"Referencias", "Vease tambien", "Enlaces externos" y hasta un "</center>"
de HTML crudo. Aprendio el registro de un articulo, no el de una persona
que responde.

FineWeb-2 es texto web filtrado y deduplicado: foros, tutoriales, blogs,
opiniones, recetas. Gente escribiendo como escribe la gente. El espanol
son 146 archivos, unos 0.64 TB, del orden de 150,000 millones de tokens
-- cien veces mas de lo que tenemos. Se baja solo la porcion que haga
falta.

Cada archivo se procesa y SE BORRA antes de bajar el siguiente, para que
el pico de disco sea el de un archivo (~5 GB) y no el de todos.
"""

import argparse
import os
import shutil
import time
import urllib.request

import pyarrow.parquet as pq

from text_cleaning import limpiar_texto

BASE = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/"
        "data/spa_Latn/train/000_{:05d}.parquet")
N_ARCHIVOS_DISPONIBLES = 146


def espacio_libre_gb(ruta: str) -> float:
    return shutil.disk_usage(os.path.dirname(os.path.abspath(ruta)) or ".").free / 1e9


def descargar(url: str, destino: str) -> None:
    if os.path.exists(destino):
        print(f"    (ya estaba descargado)", flush=True)
        return
    t0 = time.time()
    urllib.request.urlretrieve(url, destino + ".parcial")
    os.replace(destino + ".parcial", destino)
    mb = os.path.getsize(destino) / 1e6
    print(f"    bajado: {mb:,.0f} MB en {time.time()-t0:.0f}s "
          f"({mb/max(time.time()-t0, 1):.1f} MB/s)", flush=True)


def procesar(ruta: str, salida, args) -> tuple[int, int, int]:
    """Vuelca el texto util del archivo, leyendolo por trozos.

    No se carga entero en memoria: un parquet de 5 GB comprimido pasaria
    de los 12 GB al descomprimirse.
    """
    guardados = descartados = caracteres = 0
    f = pq.ParquetFile(ruta)
    columnas = ["text", "language_score", "minhash_cluster_size"]
    for lote in f.iter_batches(batch_size=20_000, columns=columnas):
        textos = lote.column("text").to_pylist()
        puntajes = lote.column("language_score").to_pylist()
        copias = lote.column("minhash_cluster_size").to_pylist()
        trozo = []
        for texto, puntaje, n_copias in zip(textos, puntajes, copias):
            # language_score: que tan seguro esta el detector de que es
            # espanol. Por debajo de 0.9 suele ser mezcla o catalan/gallego.
            # minhash_cluster_size: cuantas copias casi identicas hay en la
            # web. Los muy repetidos son plantillas, avisos legales y spam.
            if puntaje < args.min_idioma or (n_copias or 0) > args.max_copias:
                descartados += 1
                continue
            limpio = limpiar_texto(texto).strip()
            if len(limpio) < args.min_caracteres:
                descartados += 1
                continue
            trozo.append(limpio)
            guardados += 1
            caracteres += len(limpio)
        if trozo:
            salida.write("\n\n".join(trozo) + "\n\n")
    return guardados, descartados, caracteres


def main():
    p = argparse.ArgumentParser(description="Descarga texto web en espanol de FineWeb-2.")
    p.add_argument("--archivos", type=int, default=2,
                   help=f"Cuantos de los {N_ARCHIVOS_DISPONIBLES} archivos bajar "
                        "(cada uno ~4.9 GB comprimido, ~8-10 GB de texto).")
    p.add_argument("--desde", type=int, default=0, help="Primer archivo (para continuar despues).")
    p.add_argument("--salida", default="atlas_lumerak/data/fineweb_es.txt")
    p.add_argument("--tmp", default="atlas_lumerak/data/_tmp_fineweb")
    p.add_argument("--min_idioma", type=float, default=0.9)
    p.add_argument("--max_copias", type=int, default=100)
    p.add_argument("--min_caracteres", type=int, default=200)
    p.add_argument("--reserva_gb", type=float, default=15.0,
                   help="Espacio libre por debajo del cual se detiene, para no llenar el disco.")
    args = p.parse_args()

    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)

    libre = espacio_libre_gb(args.salida)
    print(f"Espacio libre en disco: {libre:.1f} GB")
    if libre < args.reserva_gb + 6:
        print(f"Poco espacio. Se necesita al menos {args.reserva_gb + 6:.0f} GB. Aborta.")
        return

    # Modo 'a': si se corta la luz, se relanza con --desde y se sigue anadiendo.
    modo = "a" if args.desde > 0 and os.path.exists(args.salida) else "w"
    total_g = total_d = total_c = 0
    t_inicio = time.time()

    with open(args.salida, modo, encoding="utf-8") as salida:
        for i in range(args.desde, min(args.desde + args.archivos, N_ARCHIVOS_DISPONIBLES)):
            libre = espacio_libre_gb(args.salida)
            if libre < args.reserva_gb:
                print(f"\nQuedan {libre:.1f} GB libres, por debajo de la reserva. Se detiene aqui.")
                print(f"Para continuar mas adelante: --desde {i}")
                break

            print(f"\n[archivo {i} de {args.desde + args.archivos - 1}] "
                  f"({libre:.1f} GB libres)", flush=True)
            ruta = os.path.join(args.tmp, f"fineweb_{i:05d}.parquet")
            descargar(BASE.format(i), ruta)

            t0 = time.time()
            g, d, c = procesar(ruta, salida, args)
            os.remove(ruta)   # borrar antes de bajar el siguiente
            total_g += g; total_d += d; total_c += c
            print(f"    {g:,} documentos guardados, {d:,} descartados "
                  f"({d/(g+d)*100:.0f}%), {c/1e9:.2f} mil millones de caracteres "
                  f"en {time.time()-t0:.0f}s", flush=True)
            print(f"    ACUMULADO: {total_c/1e9:.2f} mil millones de caracteres "
                  f"(~{total_c/3.9/1e9:.2f} mil millones de tokens)", flush=True)

    print(f"\nListo en {(time.time()-t_inicio)/60:.0f} minutos.")
    print(f"  documentos: {total_g:,} | descartados: {total_d:,}")
    print(f"  caracteres: {total_c:,}  (~{total_c/3.9/1e6:,.0f} millones de tokens)")
    print(f"  archivo: {args.salida} ({os.path.getsize(args.salida)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
