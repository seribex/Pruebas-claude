"""
Un chat en la terminal para conversar con Atlas Lumerak.

Usa el mismo formato de turnos ("Usuario: ...\nAtlas: ...\n\n") con el que
se preparan los datos de conversacion (ver prepare_oasst2.py), para que el
modelo reconozca cuando le toca responder.

Ademas, guarda cada intercambio en un archivo con tu calificacion de si la
respuesta fue util. Esas conversaciones marcadas como utiles son la materia
prima para seguir mejorando a Atlas mas adelante.
"""

import argparse
import json
import time

import torch

from checkpoint_utils import cargar_modelo

STOP_MARKER = "\nUsuario:"


def main():
    parser = argparse.ArgumentParser(description="Chatea con un Atlas Lumerak ya entrenado.")
    parser.add_argument("--checkpoint", default="atlas_lumerak/checkpoints/atlas_lumerak.pt")
    parser.add_argument("--vocab", default=None,
                        help="Tokenizador (por defecto: el que indique el checkpoint, en su misma carpeta).")
    parser.add_argument("--response_length", type=int, default=200)
    parser.add_argument("--log", default="atlas_lumerak/data/chat_log.jsonl")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40, help="0 para desactivar")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = cargar_modelo(args.checkpoint, device, args.vocab)
    block_size = config["block_size"]

    print("=== Atlas Lumerak ===")
    print(f"(memoria de hasta {block_size} tokens -- escribe 'salir' para terminar)\n")

    contexto = ""
    while True:
        try:
            entrada = input("Tu: ")
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if entrada.strip().lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        contexto += f"Usuario: {entrada}\nAtlas:"

        # Se codifica todo el contexto y se recortan los ultimos
        # block_size tokens: es lo que el modelo alcanza a "ver".
        ids = tokenizer.encode(contexto)[-block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        salida = model.generate(
            idx,
            max_new_tokens=args.response_length,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )[0].tolist()

        crudo = tokenizer.decode(salida[len(ids):])
        # El modelo no tiene una senal explicita de "aqui termina mi
        # respuesta", asi que se corta si empieza a inventar por su
        # cuenta el siguiente turno del usuario.
        respuesta = crudo.split(STOP_MARKER)[0].strip()
        print(f"Atlas: {respuesta}\n")

        contexto += f" {respuesta}\n\n"

        util = input("¿Fue util esta respuesta? (s/n, Enter para omitir): ").strip().lower()
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "usuario": entrada,
                "atlas": respuesta,
                "util": util == "s" if util in ("s", "n") else None,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
