"""
Un chat simple en la terminal para conversar con Atlas Lumerak.

Importante: el modelo actual solo aprendio a "continuar texto" (Quijote +
Wikipedia), no a conversar -- no tiene un entrenamiento especifico de
dialogo todavia (eso es el siguiente paso del proyecto). Asi que por
ahora, en vez de "responder" a lo que escribas, va a intentar continuar
el texto como si tu mensaje fuera el inicio de un parrafo. Sirve para
probar el modelo mientras tanto, y esta misma herramienta la vamos a
seguir usando cuando ya sepa conversar de verdad.

Otra limitacion real para tener en cuenta: el modelo solo "recuerda"
hasta BLOCK_SIZE caracteres hacia atras (lo que se configuro al
entrenarlo). Si la conversacion se alarga mas que eso, las partes mas
viejas simplemente dejan de ser visibles para el modelo.
"""

import argparse

import torch

from tokenizer import CharTokenizer
from model import TransformerLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Chatea con un Atlas Lumerak ya entrenado.")
    parser.add_argument("--checkpoint", default="atlas_lumerak/checkpoints/atlas_lumerak.pt")
    parser.add_argument("--vocab", default="atlas_lumerak/checkpoints/vocab.json")
    parser.add_argument("--response_length", type=int, default=200)
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

        context += user_input + "\n"

        # Solo codificamos los caracteres que el modelo puede "ver"
        # (los ultimos block_size), y solo los que existen en su
        # vocabulario -- cualquier simbolo desconocido se descarta.
        known_chars = [c for c in context[-block_size:] if c in tokenizer.char_to_id]
        idx = torch.tensor([tokenizer.encode("".join(known_chars))], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=args.response_length)[0].tolist()

        respuesta = tokenizer.decode(out[idx.shape[1]:])
        print(f"Atlas: {respuesta}\n")

        context += respuesta + "\n"


if __name__ == "__main__":
    main()
