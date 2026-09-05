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

import numpy as np
import torch

from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer
from model import TransformerLanguageModel


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # .long() aca y no en todo el corpus: la capa de embeddings exige
    # enteros de 64 bits, pero solo hace falta convertir este lote.
    return x.long().to(device), y.long().to(device)


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
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="Cada cuantos pasos guardar el progreso (0 para desactivar).")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--tie_weights", action="store_true",
                        help="Compartir la tabla de entrada y la de salida (recomendado en modelos nuevos).")
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Ruta a un checkpoint existente para seguir entrenando en vez de empezar de cero.",
    )
    parser.add_argument(
        "--resume_vocab",
        default=None,
        help="Vocabulario del checkpoint a continuar (por defecto: vocab.json junto al checkpoint).",
    )
    parser.add_argument(
        "--tokens",
        default=None,
        help="Archivo .npy con el corpus ya codificado por prepare_tokens.py (modo BPE).",
    )
    parser.add_argument(
        "--bpe",
        default=None,
        help="Tokenizador BPE (por defecto: se deduce del nombre del archivo de tokens).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo detectado: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # cudnn.benchmark deja que la GPU elija automaticamente el algoritmo
        # mas rapido para el tamano de datos que le estamos dando.
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(42)

    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location="cpu")
        print(f"Continuando entrenamiento desde: {args.resume_from}")

    # --- Tokenizador y datos ---
    if args.tokens:
        # Modo BPE: el corpus ya viene convertido a numeros por
        # prepare_tokens.py, asi que no hay que procesar texto aca.
        ruta_bpe = args.bpe or args.tokens.replace("_tokens.bin", "_bpe.json").replace("_tokens.npy", "_bpe.json")
        tokenizer = BPETokenizer.load(ruta_bpe)
        crudo = (
            np.fromfile(args.tokens, dtype=np.uint16)
            if args.tokens.endswith(".bin")
            else np.load(args.tokens)
        )
        # int32 y no int64: con miles de millones de tokens, la diferencia
        # es de 6 GB de RAM contra 12 GB. Cada lote se convierte a int64
        # (lo que exige la capa de embeddings) recien en get_batch, donde
        # son unos pocos miles de numeros.
        data = torch.from_numpy(crudo.astype(np.int32))
        del crudo
        info_tokenizador = {"tipo": "bpe", "archivo": os.path.basename(ruta_bpe)}
        print(f"Tokens pre-codificados: {args.tokens} ({len(data):,} tokens)")
        print(f"Tokenizador BPE: {ruta_bpe} ({tokenizer.vocab_size:,} simbolos)")
    else:
        # Modo caracteres (el original).
        text = ""
        for path in args.data:
            with open(path, encoding="utf-8") as f:
                text += f.read() + "\n\n"
        print(f"Archivos combinados: {', '.join(args.data)}")

        if resume_ckpt is not None:
            vocab_path = args.resume_vocab or os.path.join(os.path.dirname(args.resume_from), "vocab.json")
            tokenizer = CharTokenizer.load(vocab_path)
            # El checkpoint que se continua tiene un vocabulario fijo: si
            # los datos nuevos traen caracteres que antes no existian, se
            # descartan (agrandar el vocabulario cambiaria la forma
            # interna del modelo y los pesos ya no encajarian).
            texto_filtrado = "".join(c for c in text if c in tokenizer.char_to_id)
            descartados = len(text) - len(texto_filtrado)
            if descartados:
                print(f"Aviso: se descartaron {descartados} caracteres fuera del vocabulario original.")
            text = texto_filtrado
        else:
            tokenizer = CharTokenizer(text)

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        info_tokenizador = {"tipo": "char", "archivo": "vocab.json"}
        print(f"Corpus: {len(text):,} caracteres | vocabulario: {tokenizer.vocab_size} simbolos")

    # --- Arquitectura ---
    if resume_ckpt is not None:
        # Debe ser identica a la del checkpoint, no la de la linea de comandos.
        model_config = resume_ckpt["config"]
        print(f"Configuracion heredada del checkpoint: {model_config}")
    else:
        model_config = {
            "vocab_size": tokenizer.vocab_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "block_size": args.block_size,
            "tie_weights": args.tie_weights,
        }

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Entrenamiento: {len(train_data):,} tokens | validacion: {len(val_data):,} tokens")

    # La tasa de aprendizaje correcta depende de si el modelo es nuevo o si
    # ya esta entrenado. Continuar entrenando con la misma tasa "alta" que
    # se usa para empezar de cero puede sacudir demasiado los pesos ya
    # aprendidos (lo vimos en la practica: la calidad del chat empeoro
    # tras continuar con la tasa por defecto). Practica estandar de
    # "fine-tuning": 5-10 veces mas baja que el entrenamiento original.
    if args.lr is None:
        args.lr = 3e-5 if resume_ckpt is not None else 3e-4
        print(f"Tasa de aprendizaje automatica: {args.lr} ({'continuando' if resume_ckpt is not None else 'desde cero'})")

    # A partir de aca, el tamano de contexto SIEMPRE es el de model_config
    # (fijo por la arquitectura), nunca el de --block_size en la linea de
    # comandos -- eso evita que, al continuar un entrenamiento, se arme un
    # lote mas largo de lo que el modelo puede aceptar.
    block_size = model_config["block_size"]

    model = TransformerLanguageModel(**model_config).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros del modelo: {n_params:,}")

    # Precision mixta: en GPU, hace la mayoria de las cuentas en 16 bits
    # (bfloat16) en vez de 32 -- usa el hardware especializado ("Tensor
    # Cores") de las GPUs modernas para acelerar el entrenamiento, con
    # perdida de precision insignificante para esto.
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda"))

    if device == "cuda":
        # torch.compile() en si nunca falla -- solo "envuelve" el modelo.
        # El error real (por ejemplo, si falta Triton en el sistema) recien
        # aparece la PRIMERA VEZ que el modelo compilado se usa de verdad.
        # Por eso forzamos un paso de prueba aca dentro, en vez de solo
        # envolver la llamada a torch.compile en el try/except.
        compiled_model = torch.compile(model)
        try:
            xb_test, yb_test = get_batch(train_data, block_size, args.batch_size, device)
            with autocast_ctx:
                compiled_model(xb_test, yb_test)
            model = compiled_model
            print("torch.compile activado (el primer paso tardara un poco mas mientras compila).")
        except Exception as e:
            print(f"Aviso: no se pudo activar torch.compile ({type(e).__name__}: {e}), se sigue sin el.")

    # El "decaimiento de pesos" empuja los parametros hacia cero para que el
    # modelo no dependa demasiado de valores extremos. Pero solo tiene
    # sentido en las matrices grandes: aplicarselo a los sesgos y a los
    # parametros de LayerNorm (que ajustan escala, no representan
    # conocimiento) es contraproducente. Por eso van en dos grupos.
    con_decaimiento = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    sin_decaimiento = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    grupos = [
        {"params": con_decaimiento, "weight_decay": args.weight_decay},
        {"params": sin_decaimiento, "weight_decay": 0.0},
    ]
    print(f"Decaimiento de pesos {args.weight_decay} sobre {len(con_decaimiento)} matrices; "
          f"0.0 sobre {len(sin_decaimiento)} sesgos/LayerNorm")

    # El optimizador "fusionado" actualiza todos los parametros en una sola
    # operacion en la GPU en vez de una por una -- mismo resultado, mas rapido.
    try:
        optimizer = torch.optim.AdamW(grupos, lr=args.lr, fused=(device == "cuda"))
    except (RuntimeError, TypeError) as e:
        print(f"Aviso: no se pudo usar el optimizador fusionado ({e}), usando el normal.")
        optimizer = torch.optim.AdamW(grupos, lr=args.lr)

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
    # Se guarda el tokenizador junto al modelo: sin el, los pesos son
    # inutiles (no habria forma de traducir texto a numeros y de vuelta).
    tokenizer.save(os.path.join(args.out_dir, info_tokenizador["archivo"]))

    checkpoint_path = os.path.join(args.out_dir, "atlas_lumerak.pt")

    def guardar(ruta: str) -> None:
        estado = {
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "tokenizador": info_tokenizador,
        }
        # Se escribe primero a un archivo temporal y recien despues se
        # renombra: si el proceso muere a mitad de la escritura, el
        # checkpoint anterior queda intacto en vez de quedar corrupto.
        torch.save(estado, ruta + ".tmp")
        os.replace(ruta + ".tmp", ruta)

    start_time = time.time()
    for step in range(args.steps):
        xb, yb = get_batch(train_data, block_size, args.batch_size, device)
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
            losses = estimate_loss(model, train_data, val_data, block_size, args.batch_size, device, autocast_ctx)
            elapsed = time.time() - start_time
            print(
                f"paso {step:5d}/{args.steps} | "
                f"perdida entrenamiento {losses['train']:.4f} | "
                f"perdida validacion {losses['val']:.4f} | "
                f"{elapsed:.0f}s transcurridos"
            )

        # Guardado intermedio: en una corrida de muchas horas, perder todo
        # por un corte de luz o un error a ultimo momento no es aceptable.
        if args.save_interval > 0 and step > 0 and step % args.save_interval == 0:
            guardar(os.path.join(args.out_dir, "atlas_lumerak_parcial.pt"))
            print(f"    (progreso guardado en el paso {step:,})", flush=True)

    # Nunca sobrescribir en silencio un checkpoint anterior: si ya existe
    # uno (por ejemplo, el que se esta usando como base con --resume_from),
    # se respalda con la fecha y hora antes de guardar el nuevo. Asi, si
    # esta corrida resulta peor (como paso una vez), el anterior no se
    # pierde.
    if os.path.exists(checkpoint_path):
        backup_path = checkpoint_path.replace(".pt", f"_backup_{time.strftime('%Y%m%d_%H%M%S')}.pt")
        os.replace(checkpoint_path, backup_path)
        print(f"Checkpoint anterior respaldado en: {backup_path}")

    guardar(checkpoint_path)
    parcial = os.path.join(args.out_dir, "atlas_lumerak_parcial.pt")
    if os.path.exists(parcial):
        os.remove(parcial)  # ya no hace falta: el definitivo esta completo
    print(f"\nModelo guardado en: {checkpoint_path}")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(start, max_new_tokens=300)[0].tolist()
    print("\n--- Muestra generada ---")
    print(tokenizer.decode(sample))


if __name__ == "__main__":
    main()
