"""
PASO 6 (datos): descarga y prepara conversaciones reales en espanol para
ensenarle a Atlas Lumerak a dialogar, no solo a completar texto.

Fuente: OpenAssistant/oasst2, un dataset de conversaciones humano-asistente
creado por miles de voluntarios especificamente para entrenar asistentes
conversacionales (licencia Apache 2.0, uso libre). De sus 135,174 mensajes
en 32+ idiomas, filtramos los de espanol que pasaron revision de calidad
(no eliminados, con revision positiva), y armamos pares pregunta-respuesta
siguiendo la relacion padre-hijo de cada arbol de conversacion.

El resultado se guarda en el mismo formato que usa chat.py para las
conversaciones ("Usuario: ...\nAtlas: ...\n\n"), para que el modelo
aprenda ese patron de turnos durante el entrenamiento.
"""

import argparse
import unicodedata
import urllib.request

import pandas as pd

TRAIN_URL = "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
VAL_URL = "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"


def keep_char(c: str) -> bool:
    if c.isascii():
        return True
    # Ojo: algunos simbolos/emojis (ej. "LATIN CROSS", letras encerradas en
    # circulos) tienen "LATIN" en su nombre Unicode sin ser letras reales
    # -- por eso exigimos ademas que la categoria sea de tipo "Letra" (L*).
    cat = unicodedata.category(c)
    if cat.startswith("L") and "LATIN" in unicodedata.name(c, ""):
        return True
    return c in "¡¿«»—–…‘’“”"


def download(url: str, path: str) -> None:
    urllib.request.urlretrieve(url, path)


def calidad_ok(row) -> bool:
    return (row["deleted"] == False) and (row["review_result"] == True)  # noqa: E712


def main():
    parser = argparse.ArgumentParser(description="Prepara el corpus de conversacion oasst2 en espanol.")
    parser.add_argument("--out", default="atlas_lumerak/data/conversaciones_oasst2.txt")
    parser.add_argument("--train_parquet", default="/tmp/oasst2_train.parquet")
    parser.add_argument("--val_parquet", default="/tmp/oasst2_validation.parquet")
    args = parser.parse_args()

    print("Descargando oasst2 (~66 MB)...")
    download(TRAIN_URL, args.train_parquet)
    download(VAL_URL, args.val_parquet)

    df = pd.concat(
        [pd.read_parquet(args.train_parquet), pd.read_parquet(args.val_parquet)],
        ignore_index=True,
    )
    by_id = df.set_index("message_id")

    asistente = df[(df["role"] == "assistant") & (df["lang"] == "es")]

    pares = []
    for _, msg in asistente.iterrows():
        if not calidad_ok(msg):
            continue
        parent_id = msg["parent_id"]
        if parent_id not in by_id.index:
            continue
        padre = by_id.loc[parent_id]
        if padre["role"] != "prompter" or padre["lang"] != "es" or not calidad_ok(padre):
            continue
        pregunta = str(padre["text"]).strip()
        respuesta = str(msg["text"]).strip()
        if len(pregunta) < 3 or len(respuesta) < 3:
            continue
        pares.append((pregunta, respuesta))

    print(f"Pares pregunta-respuesta en espanol, de buena calidad: {len(pares)}")

    texto = "".join(f"Usuario: {p}\nAtlas: {r}\n\n" for p, r in pares)
    texto_limpio = "".join(c for c in texto if keep_char(c))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(texto_limpio)

    print(f"Guardado en {args.out}: {len(texto_limpio):,} caracteres")


if __name__ == "__main__":
    main()
