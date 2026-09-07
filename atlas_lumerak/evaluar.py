"""
Mide que tan bueno es un modelo guardado, sin entrenar nada.

Sirve para responder la pregunta que importa cuando una corrida se
interrumpe: no "en que paso quedo" sino "que tan bueno es lo que tengo".

Reporta:
  - Perdida sobre datos que el modelo nunca vio (el numero comparable
    entre corridas)
  - Perplejidad: la misma medida en una escala mas intuitiva -- "entre
    cuantas opciones duda el modelo, en promedio, antes de elegir la
    siguiente pieza de texto". Menos es mejor.
  - Una muestra de texto generado, para ver la calidad con los ojos y no
    solo con un numero.
"""

import argparse
import math

import numpy as np
import torch

from checkpoint_utils import cargar_modelo


def main():
    parser = argparse.ArgumentParser(description="Evalua un checkpoint de Atlas Lumerak.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", required=True, help="Los mismos tokens con los que se entreno.")
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--lotes", type=int, default=200, help="Cuantos lotes promediar.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--muestra", type=int, default=300)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = cargar_modelo(args.checkpoint, device, args.vocab)
    block_size = config["block_size"]
    print(f"Modelo: {sum(p.numel() for p in model.parameters()):,} parametros | {config}")

    crudo = (np.fromfile(args.tokens, dtype=np.uint16)
             if args.tokens.endswith(".bin") else np.load(args.tokens))
    data = torch.from_numpy(crudo.astype(np.int32))
    del crudo

    # Se evalua sobre el ultimo 10%: exactamente la misma porcion que
    # train.py aparta para validacion, o sea texto que el modelo no vio.
    val = data[int(0.9 * len(data)):]
    print(f"Evaluando sobre {len(val):,} tokens de validacion ({args.lotes} lotes)\n")

    torch.manual_seed(0)
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda"))

    perdidas = []
    with torch.no_grad():
        for _ in range(args.lotes):
            ix = torch.randint(len(val) - block_size, (args.batch_size,))
            x = torch.stack([val[i:i + block_size] for i in ix]).long().to(device)
            y = torch.stack([val[i + 1:i + block_size + 1] for i in ix]).long().to(device)
            with autocast_ctx:
                _, loss = model(x, y)
            perdidas.append(loss.item())

    media = sum(perdidas) / len(perdidas)
    print(f"PERDIDA DE VALIDACION: {media:.4f}")
    print(f"PERPLEJIDAD: {math.exp(media):.1f}  (duda entre ~{math.exp(media):.0f} opciones en promedio)")
    print(f"Referencia: un modelo sin entrenar daria {math.log(config['vocab_size']):.2f} "
          f"de perdida y {config['vocab_size']:,} de perplejidad\n")

    ids = tokenizer.encode(args.prompt) if args.prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    salida = model.generate(idx, max_new_tokens=args.muestra, temperature=0.8, top_k=40)[0].tolist()
    print("--- Muestra generada ---")
    print(tokenizer.decode(salida))


if __name__ == "__main__":
    main()
