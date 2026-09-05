"""
Un chat en la terminal para conversar con Atlas Lumerak.

Usa el mismo formato de turnos ("Usuario: ...\nAtlas: ...\n\n") que se
usa para preparar los datos de conversacion (ver prepare_oasst2.py), asi
que una vez que el modelo se entrene con esos datos, va a reconocer este
patron y responder como dialogo real. Si todavia estas usando un modelo
que solo vio Quijote + Wikipedia (sin conversaciones), esto se va a
sentir mas como autocompletar texto que como una charla -- es esperado.

Ademas, guarda cada intercambio en un archivo, con tu calificacion de si
la respuesta fue util o no. Esas conversaciones marcadas como utiles son
las que despues se pueden sumar como datos nuevos para seguir mejorando
a Atlas Lumerak.
"""

import argparse
import json
import time

import torch

from tokenizer import CharTokenizer
from model import TransformerLanguageModel

STOP_MARKER = "\nUsuario:"


def main():
    parser = argparse.ArgumentParser(description="Chatea con un Atlas Lumerak ya entrenado.")
    parser.add_argument("--checkpoint", default="atlas_lumerak/checkpoints/atlas_lumerak.pt")
    parser.add_argument("--vocab", default="atlas_lumerak/checkpoints/vocab.json")
    parser.add_argument("--response_length", type=int, default=200)
    parser.add_argument("--log", default="atlas_lumerak/data/chat_log.jsonl")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40, help="0 para desactivar")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CharTokenizer.load(args.vocab)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = TransformerLanguageModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    block_size = checkpoint["config"]["block_size"]

    print("=== Atlas Lumerak ===")
    print(f"(memoria de hasta {block_size} caracteres -- escribe 'salir' para terminar)\n")

    context = ""
    while True:
        try:
            user_input = input("Tu: ")
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if user_input.strip().lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        context += f"Usuario: {user_input}\nAtlas:"

        # Solo codificamos los caracteres que el modelo puede "ver"
        # (los ultimos block_size), y solo los que existen en su
        # vocabulario -- cualquier simbolo desconocido se descarta.
        known_chars = [c for c in context[-block_size:] if c in tokenizer.char_to_id]
        idx = torch.tensor([tokenizer.encode("".join(known_chars))], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.response_length,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
            )[0].tolist()

        crudo = tokenizer.decode(out[idx.shape[1]:])
        # El modelo no tiene una señal explicita de "aqui termino mi
        # respuesta", asi que cortamos si empieza a inventar el
        # siguiente turno de "Usuario:" por su cuenta.
        respuesta = crudo.split(STOP_MARKER)[0].strip()
        print(f"Atlas: {respuesta}\n")

        context += f" {respuesta}\n\n"

        util = input("¿Fue util esta respuesta? (s/n, Enter para omitir): ").strip().lower()
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "usuario": user_input,
                "atlas": respuesta,
                "util": util == "s" if util in ("s", "n") else None,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
