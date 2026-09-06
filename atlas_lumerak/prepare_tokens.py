"""
PASO 8: Convertir el corpus completo a numeros, una sola vez.

Entrena el tokenizador BPE y despues codifica TODO el texto a una lista
de numeros que se guarda en disco. Sin esto, cada corrida de
entrenamiento tendria que volver a procesar gigas de texto desde cero.

Se corre una vez; despues train.py carga el archivo al instante.

Dos decisiones importantes por el tamano del corpus (varios GB):

1. El BPE se entrena sobre una MUESTRA, no sobre todo. Para aprender que
   fragmentos son frecuentes en espanol alcanza con decenas de millones
   de caracteres. La muestra se toma de varias partes del archivo, no
   solo del principio, para que sea representativa.

2. Todo se procesa por partes y se va escribiendo a disco sobre la
   marcha. Cargar 5 GB de texto de golpe usaria mas de 10 GB de memoria
   (Python gasta 2 bytes por caracter cuando hay acentos y comillas
   tipograficas), y eso antes de siquiera empezar a codificar.
"""

import argparse
import os
import time

import numpy as np

from bpe_tokenizer import BPETokenizer

CHUNK = 20_000_000  # caracteres de texto que se codifican de una vez


def tomar_muestra(rutas: list[str], total_chars: int) -> str:
    """Junta texto de varios puntos repartidos del corpus, no solo del inicio."""
    tamanos = [os.path.getsize(r) for r in rutas]
    total_bytes = sum(tamanos)
    partes = []

    for ruta, tam in zip(rutas, tamanos):
        # A cada archivo le toca una porcion proporcional a su tamano.
        cuota = int(total_chars * tam / total_bytes)
        if cuota <= 0:
            continue
        n_trozos = 5
        por_trozo = cuota // n_trozos
        with open(ruta, encoding="utf-8", errors="ignore") as f:
            for k in range(n_trozos):
                f.seek(int(tam * k / n_trozos))
                f.readline()  # descartar la linea cortada por el salto
                partes.append(f.read(por_trozo))
    return "\n".join(partes)


def codificar_archivos(rutas: list[str], tok: BPETokenizer, ruta_salida: str) -> int:
    """Codifica los archivos por partes, escribiendo a disco sobre la marcha."""
    total_tokens = 0
    total_chars = 0
    tamano_total = sum(os.path.getsize(r) for r in rutas)

    with open(ruta_salida, "wb") as salida:
        for ruta in rutas:
            with open(ruta, encoding="utf-8") as f:
                sobrante = ""
                while True:
                    trozo = f.read(CHUNK)
                    if not trozo:
                        break
                    trozo = sobrante + trozo

                    # No cortar a mitad de palabra: se guarda lo que sigue
                    # despues del ultimo salto de linea para el proximo trozo.
                    corte = trozo.rfind("\n")
                    if corte == -1:
                        sobrante = ""
                    else:
                        sobrante = trozo[corte + 1:]
                        trozo = trozo[: corte + 1]

                    ids = np.array(tok.encode(trozo), dtype=np.uint16)
                    ids.tofile(salida)
                    total_tokens += len(ids)
                    total_chars += len(trozo)
                    print(
                        f"  {total_chars / tamano_total * 100:5.1f}% | "
                        f"{total_chars:,} caracteres -> {total_tokens:,} tokens",
                        flush=True,
                    )

                if sobrante:
                    ids = np.array(tok.encode(sobrante), dtype=np.uint16)
                    ids.tofile(salida)
                    total_tokens += len(ids)
                    total_chars += len(sobrante)

    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="Entrena BPE y pre-codifica el corpus.")
    parser.add_argument("--data", nargs="+", required=True,
                        help="Archivos a convertir en tokens.")
    parser.add_argument("--bpe_data", nargs="+", default=None,
                        help="Archivos con los que APRENDER el tokenizador (por defecto, los mismos "
                             "de --data). Util para que el tokenizador conozca todos los registros "
                             "del proyecto aunque esta corrida solo codifique uno de ellos.")
    parser.add_argument("--reuse_bpe", default=None,
                        help="Reutilizar un tokenizador ya entrenado en vez de aprender uno nuevo. "
                             "Imprescindible para que dos corpus queden en el mismo 'idioma' de tokens.")
    parser.add_argument("--out_dir", default="atlas_lumerak/data")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--train_sample_chars", type=int, default=50_000_000,
                        help="Cuanto texto usar para APRENDER las fusiones BPE.")
    parser.add_argument("--nombre", default="corpus")
    args = parser.parse_args()

    if args.vocab_size > 65535:
        raise ValueError("vocab_size no puede pasar de 65535 (los tokens se guardan como uint16).")

    os.makedirs(args.out_dir, exist_ok=True)
    ruta_tok = args.reuse_bpe or os.path.join(args.out_dir, f"{args.nombre}_bpe.json")

    if os.path.exists(ruta_tok):
        print(f"Reutilizando tokenizador existente: {ruta_tok}")
        tok = BPETokenizer.load(ruta_tok)
    else:
        fuentes_bpe = args.bpe_data or args.data
        print(f"Tomando muestra de {args.train_sample_chars:,} caracteres de: {', '.join(fuentes_bpe)}")
        muestra = tomar_muestra(fuentes_bpe, args.train_sample_chars)
        print(f"Muestra obtenida: {len(muestra):,} caracteres\n")

        print("Entrenando BPE...")
        t0 = time.time()
        tok = BPETokenizer()
        tok.train(muestra, vocab_size=args.vocab_size)
        print(f"BPE entrenado en {time.time() - t0:.0f}s")
        del muestra

        tok.save(ruta_tok)
        print(f"Tokenizador guardado en {ruta_tok}\n")

    ruta_tokens = os.path.join(args.out_dir, f"{args.nombre}_tokens.bin")
    print("Codificando el corpus completo (por partes, sin cargarlo entero en memoria)...")
    t0 = time.time()
    total_tokens = codificar_archivos(args.data, tok, ruta_tokens)

    print(f"\nCodificado en {time.time() - t0:.0f}s")
    print(f"Tokens: {total_tokens:,}")
    print(f"Guardado en {ruta_tokens} ({os.path.getsize(ruta_tokens) / 1e9:.2f} GB)")
    print(f"\nPara entrenar:\n  python atlas_lumerak/train.py --tokens {ruta_tokens}")


if __name__ == "__main__":
    main()
