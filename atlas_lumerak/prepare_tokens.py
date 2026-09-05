"""
PASO 8: Convertir el corpus completo a numeros, una sola vez.

Entrena el tokenizador BPE y despues codifica TODO el texto a una lista
de numeros que se guarda en disco (.npy). Sin esto, cada corrida de
entrenamiento tendria que volver a procesar gigas de texto desde cero.

Se corre una vez; despues train.py carga el archivo .npy al instante.

El BPE se entrena sobre una MUESTRA del corpus (no sobre los 5 GB): para
aprender que fragmentos son frecuentes en espanol alcanza y sobra con
unas decenas de millones de caracteres, y hacerlo asi baja el tiempo de
minutos-horas a un par de minutos. Despues esa tokenizacion aprendida se
aplica al corpus entero.
"""

import argparse
import os
import time

import numpy as np

from bpe_tokenizer import BPETokenizer

CHUNK = 20_000_000  # cuanto texto se codifica de una vez (en caracteres)


def leer_corpus(rutas: list[str]) -> str:
    partes = []
    for r in rutas:
        with open(r, encoding="utf-8") as f:
            partes.append(f.read())
        print(f"  leido {r}: {len(partes[-1]):,} caracteres")
    return "\n\n".join(partes)


def main():
    parser = argparse.ArgumentParser(description="Entrena BPE y pre-codifica el corpus.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out_dir", default="atlas_lumerak/data")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--train_sample_chars", type=int, default=50_000_000,
                        help="Cuanto texto usar para APRENDER las fusiones BPE.")
    parser.add_argument("--nombre", default="corpus")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Leyendo corpus...")
    texto = leer_corpus(args.data)
    print(f"Corpus total: {len(texto):,} caracteres\n")

    muestra = texto[: args.train_sample_chars]
    print(f"Entrenando BPE sobre una muestra de {len(muestra):,} caracteres...")
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(muestra, vocab_size=args.vocab_size)
    print(f"BPE entrenado en {time.time() - t0:.0f}s\n")

    ruta_tok = os.path.join(args.out_dir, f"{args.nombre}_bpe.json")
    tok.save(ruta_tok)
    print(f"Tokenizador guardado en {ruta_tok}\n")

    print("Codificando el corpus completo...")
    t0 = time.time()
    trozos = []
    pos = 0
    while pos < len(texto):
        fin = min(pos + CHUNK, len(texto))
        # No cortar a mitad de una palabra: extender hasta el proximo salto
        # de linea (salvo que ya sea el final del texto).
        if fin < len(texto):
            salto = texto.find("\n", fin)
            fin = salto + 1 if salto != -1 else len(texto)

        trozos.append(np.array(tok.encode(texto[pos:fin]), dtype=np.uint16))
        pos = fin
        print(f"  {pos:,}/{len(texto):,} caracteres ({pos / len(texto) * 100:.1f}%)", flush=True)

    tokens = np.concatenate(trozos)
    del trozos

    if tok.vocab_size > 65535:
        raise ValueError("El vocabulario no entra en uint16; usar un vocab_size menor.")

    ruta_tokens = os.path.join(args.out_dir, f"{args.nombre}_tokens.npy")
    np.save(ruta_tokens, tokens)

    print(f"\nCodificado en {time.time() - t0:.0f}s")
    print(f"Tokens: {len(tokens):,} (desde {len(texto):,} caracteres)")
    print(f"Compresion: {len(texto) / len(tokens):.2f} caracteres por token")
    print(f"Guardado en {ruta_tokens} ({os.path.getsize(ruta_tokens) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
