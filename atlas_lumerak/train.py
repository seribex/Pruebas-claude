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
import math
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
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, autocast_ctx, eval_iters=50):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            with autocast_ctx:
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
        default=[
            "atlas_lumerak/data/quijote.txt",
            "atlas_lumerak/data/wikipedia_es.txt",
            "atlas_lumerak/data/conversaciones_oasst2.txt",
        ],
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
        # cudnn.benchmark deja que la GPU elija automaticamente el algoritmo
        # mas rapido para el tamano de datos que le estamos dando.
        torch.backends.cudnn.benchmark = True

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

    # Precision mixta: en GPU, hace la mayoria de las cuentas en 16 bits
    # (bfloat16) en vez de 32 -- usa el hardware especializado ("Tensor
    # Cores") de las GPUs modernas para acelerar el entrenamiento, con
    # perdida de precision insignificante para esto.
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda"))

    if device == "cuda":
        try:
            model = torch.compile(model)
            print("torch.compile activado (el primer paso tardara un poco mas mientras compila).")
        except Exception as e:
            print(f"Aviso: no se pudo activar torch.compile ({e}), se sigue sin el.")

    # El optimizador "fusionado" actualiza todos los parametros en una sola
    # operacion en la GPU en vez de una por una -- mismo resultado, mas rapido.
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=(device == "cuda"))
    except (RuntimeError, TypeError) as e:
        print(f"Aviso: no se pudo usar el optimizador fusionado ({e}), usando el normal.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Tasa de aprendizaje: empieza baja, sube gradualmente (warmup) para no
    # desestabilizar el entrenamiento al inicio, y despues baja suavemente
    # (coseno) hacia el final -- ayuda al modelo a aprender mejor con la
    # misma cantidad de pasos, no solo a ir mas rapido por paso.
    warmup_steps = max(1, int(0.03 * args.steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.out_dir, "vocab.json"))

    start_time = time.time()
    for step in range(args.steps):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        with autocast_ctx:
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Evita que gradientes anormalmente grandes (poco frecuentes, pero
        # posibles) den un paso demasiado brusco y desestabilicen el
        # entrenamiento -- permite usar una tasa de aprendizaje mas alta
        # con seguridad.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device, autocast_ctx)
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
