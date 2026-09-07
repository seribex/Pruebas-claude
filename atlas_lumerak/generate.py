"""
Genera texto usando un modelo ya entrenado (guardado por train.py), sin
tener que volver a entrenar desde cero cada vez.
"""

import argparse

import torch

from checkpoint_utils import cargar_modelo
from inferencia import id_fin_de_turno


def main():
    parser = argparse.ArgumentParser(description="Genera texto con un Atlas Lumerak ya entrenado.")
    parser.add_argument("--checkpoint", default="atlas_lumerak/checkpoints/atlas_lumerak.pt")
    parser.add_argument("--vocab", default=None,
                        help="Tokenizador (por defecto: el que indique el checkpoint, en su misma carpeta).")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40, help="0 para desactivar")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Se queda con los pocos candidatos que junten esta probabilidad. "
                             "1.0 para desactivar.")
    parser.add_argument("--repeticion", type=float, default=1.1,
                        help="Mayor a 1.0 castiga lo que acaba de decir, para frenar bucles.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = cargar_modelo(args.checkpoint, device, args.vocab)

    start_ids = tokenizer.encode(args.prompt) if args.prompt else [0]
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    output = model.generate(
        idx,
        max_new_tokens=args.length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p < 1.0 else None,
        repetition_penalty=args.repeticion,
        stop_id=id_fin_de_turno(tokenizer),
    )[0].tolist()
    print(tokenizer.decode(output))


if __name__ == "__main__":
    main()
