"""
PASO 5: Entrenamiento real, con un dataset grande.

Este script junta todo lo anterior (tokenizador + modelo Transformer) y
lo entrena sobre un archivo de texto de verdad, no las 4 lineas de
prueba. Detecta automaticamente si hay una GPU disponible (torch.cuda)
y la usa; si no, usa CPU (mas lento, pero funciona igual para pruebas
chicas).

Guarda el resultado en un "checkpoint": un archivo con los pesos ya
entrenados del modelo, para no tener que repetir el entrenamiento cada
vez que se quiera generar texto (eso lo hace generate.py).
"""

import argparse
import os
import time

import torch

from tokenizer import CharTokenizer
from model import TransformerLanguageModel


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(description="Entrena a Atlas Lumerak sobre un corpus de texto.")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["atlas_lumerak/data/quijote.txt", "atlas_lumerak/data/wikipedia_es.txt"],
        help="Uno o mas archivos de texto; se concatenan en un solo corpus de entrenamiento.",
    )
    parser.add_argument("--out_dir", default="atlas_lumerak/checkpoints")
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo detectado: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)

    text = ""
    for path in args.data:
        with open(path, encoding="utf-8") as f:
            text += f.read() + "\n\n"
    print(f"Archivos combinados: {', '.join(args.data)}")

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Corpus: {len(text):,} caracteres | vocabulario: {tokenizer.vocab_size} simbolos")
    print(f"Entrenamiento: {len(train_data):,} caracteres | validacion: {len(val_data):,} caracteres")

    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros del modelo: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.out_dir, "vocab.json"))

    start_time = time.time()
    for step in range(args.steps):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device)
            elapsed = time.time() - start_time
            print(
                f"paso {step:5d}/{args.steps} | "
                f"perdida entrenamiento {losses['train']:.4f} | "
                f"perdida validacion {losses['val']:.4f} | "
                f"{elapsed:.0f}s transcurridos"
            )

    checkpoint_path = os.path.join(args.out_dir, "atlas_lumerak.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "block_size": args.block_size,
            },
        },
        checkpoint_path,
    )
    print(f"\nModelo guardado en: {checkpoint_path}")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(start, max_new_tokens=300)[0].tolist()
    print("\n--- Muestra generada ---")
    print(tokenizer.decode(sample))


if __name__ == "__main__":
    main()
