"""
PASO 11: Fine-tuning de instrucciones, hecho como corresponde.

Diferencias con train.py (que sigue siendo el correcto para el
pre-entrenamiento):

1) MASCARA DE PERDIDA. Solo se aprende de los tokens que dice Atlas. Los
   de "Usuario: <pregunta>" se marcan con -100 y la funcion de perdida los
   ignora por completo. En la corrida anterior no lo haciamos, y la mitad
   del gradiente se gastaba ensenandole a inventar preguntas -- que es
   exactamente lo que despues hacia al conversar.

2) LOTES ALINEADOS. Cada ejemplo del lote es una ventana preparada por
   prepare_sft.py que empieza en un "Usuario:" completo, no un recorte al
   azar del medio de una respuesta.

3) MEZCLA DE PRE-ENTRENAMIENTO. Una fraccion de los lotes viene del corpus
   original de Wikipedia. Sin esto, el modelo se especializa tanto en
   conversar que se le empieza a olvidar lo que sabia ("olvido
   catastrofico"). Es ademas un regularizador: permite entrenar bastantes
   mas pasos antes de que la validacion empiece a subir.

4) SE GUARDA EL MEJOR, NO EL ULTIMO. La corrida anterior toco su mejor
   punto en el paso 500 y empeoro despues; nos quedamos con el peor por no
   estar guardando el mejor. Ahora cada vez que la validacion mejora se
   graba una copia aparte.
"""

import argparse
import json
import math
import os
import time
import warnings

import numpy as np
import torch

from bpe_tokenizer import BPETokenizer
from model import TransformerLanguageModel

IGNORAR = -100  # valor que F.cross_entropy descarta (ver model.py)


class DatosSFT:
    """Ventanas ya alineadas y su mascara, listas para armar lotes."""

    def __init__(self, base: str):
        with open(base + "_meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        ventana = self.meta["ventana"]
        ids = np.fromfile(base + "_tokens.bin", dtype=np.uint16)
        mask = np.fromfile(base + "_mask.bin", dtype=np.uint8)
        if len(ids) != len(mask) or len(ids) % ventana != 0:
            raise ValueError("Los archivos de tokens y mascara no cuadran con el meta.json.")
        self.ids = torch.from_numpy(ids.astype(np.int32)).view(-1, ventana)
        self.mask = torch.from_numpy(mask).view(-1, ventana)

    def __len__(self):
        return self.ids.shape[0]

    def lote(self, indices, device):
        v_ids = self.ids[indices].long()
        v_mask = self.mask[indices]
        x = v_ids[:, :-1]
        y = v_ids[:, 1:].clone()
        # La mascara marca "este token hay que aprender a producirlo". Como
        # el objetivo va corrido en uno, la mascara del objetivo es la
        # misma corrida en uno.
        y[v_mask[:, 1:] == 0] = IGNORAR
        return x.to(device), y.to(device)


def quitar_prefijo_compile(estado: dict) -> dict:
    """Un modelo envuelto por torch.compile guarda sus pesos con el prefijo
    "_orig_mod." delante de cada nombre. Un checkpoint asi no se puede cargar
    en un modelo normal. Se le quita si lo trae."""
    if any(k.startswith("_orig_mod.") for k in estado):
        return {k.removeprefix("_orig_mod."): v for k, v in estado.items()}
    return estado


def lote_pre(data: torch.Tensor, block_size: int, batch_size: int, device: str,
             generador: torch.Generator | None = None):
    """Lote normal del corpus de pre-entrenamiento: sin mascara, todo cuenta.

    `generador` existe para las evaluaciones, que necesitan sacar siempre los
    mismos lotes. Tiene que ser un generador PROPIO y no la semilla global:
    reiniciar la global desde una evaluacion tambien reiniciaria la eleccion
    de lotes del entrenamiento, y el modelo pasaria el resto de la corrida
    viendo una y otra vez los mismos ejemplos sin que nada lo delate."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generador)
    x = torch.stack([data[i:i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).long()
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluar_olvido(model, pre_val, block_size, batch_size, device, autocast_ctx, lotes=20):
    """Perdida sobre Wikipedia, en el mismo 10% apartado con el que se midio
    el modelo antes de empezar. Sirve para una sola pregunta: mientras aprende
    a conversar, ¿cuanto esta olvidando de lo que sabia? Es directamente
    comparable con el numero que da evaluar.py."""
    model.eval()
    total = 0.0
    g = torch.Generator().manual_seed(0)
    for _ in range(lotes):
        x, y = lote_pre(pre_val, block_size, batch_size, device, generador=g)
        with autocast_ctx:
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / lotes


@torch.no_grad()
def evaluar(model, datos, indices_val, batch_size, device, autocast_ctx, lotes=40):
    """Perdida sobre conversaciones que el modelo no vio, contando SOLO los
    tokens de respuesta. Es el numero que de verdad dice si esta aprendiendo
    a responder."""
    model.eval()
    total, g = 0.0, torch.Generator().manual_seed(0)
    n = min(lotes, max(1, len(indices_val) // batch_size))
    for _ in range(n):
        sel = indices_val[torch.randint(len(indices_val), (batch_size,), generator=g)]
        x, y = datos.lote(sel, device)
        with autocast_ctx:
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n


def main():
    p = argparse.ArgumentParser(description="Fine-tuning conversacional de Atlas Lumerak.")
    p.add_argument("--sft", required=True,
                   help="Prefijo de los archivos de prepare_sft.py (sin _tokens.bin).")
    p.add_argument("--checkpoint", required=True, help="Modelo pre-entrenado del que partir.")
    p.add_argument("--bpe", required=True)
    p.add_argument("--tokens_pre", default=None,
                   help="Corpus de pre-entrenamiento para mezclar (evita el olvido).")
    p.add_argument("--mezcla_pre", type=float, default=0.1,
                   help="Fraccion de lotes que vienen del corpus original. 0 para desactivar. "
                        "Es un equilibrio: sirve para que no olvide lo que sabe, pero el prior "
                        "de Wikipedia es justamente lo que queremos vencer, asi que con un "
                        "corpus de instrucciones grande conviene poco, no mucho.")
    p.add_argument("--out_dir", default="atlas_lumerak/checkpoints_chat")
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--val_frac", type=float, default=0.02)
    p.add_argument("--continuar", action="store_true",
                   help="Retomar una corrida interrumpida desde <out_dir>/atlas_lumerak_parcial.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    datos = DatosSFT(args.sft)
    print(f"Corpus de instrucciones: {len(datos):,} ventanas de {datos.meta['ventana']} tokens")
    print(f"  fuentes: {datos.meta['fuentes']}")
    print(f"  tokens supervisados: {datos.meta['tokens_supervisados']:,} de "
          f"{datos.meta['tokens_totales']:,} "
          f"({datos.meta['tokens_supervisados'] / datos.meta['tokens_totales'] * 100:.1f}%)")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    block_size = config["block_size"]
    if block_size != datos.meta["block_size"]:
        raise ValueError(
            f"El modelo usa contexto de {block_size} tokens y el corpus se preparo para "
            f"{datos.meta['block_size']}. Vuelve a correr prepare_sft.py con "
            f"--block_size {block_size}.")
    tok = BPETokenizer.load(args.bpe)
    if tok.vocab_size != config["vocab_size"]:
        raise ValueError(f"El tokenizador tiene {tok.vocab_size} simbolos y el modelo espera "
                         f"{config['vocab_size']}. No son el mismo.")

    model = TransformerLanguageModel(**config).to(device)
    model.load_state_dict(quitar_prefijo_compile(ckpt["model_state_dict"]))
    print(f"Modelo cargado: {sum(x.numel() for x in model.parameters()):,} parametros | {config}")

    # Separacion honesta: las ventanas de validacion nunca se entrenan.
    g = torch.Generator().manual_seed(7)
    orden = torch.randperm(len(datos), generator=g)
    n_val = max(args.batch_size, int(args.val_frac * len(datos)))
    idx_val, idx_train = orden[:n_val], orden[n_val:]
    print(f"Entrenamiento: {len(idx_train):,} ventanas | validacion: {len(idx_val):,}")

    pre_train = pre_val = None
    if args.tokens_pre:
        crudo = np.fromfile(args.tokens_pre, dtype=np.uint16)
        datos_pre = torch.from_numpy(crudo.astype(np.int32))
        del crudo
        # El mismo corte 90/10 que uso train.py. Es importante que la mezcla
        # salga solo del 90% de entrenamiento: si entrenaramos sobre el 10%
        # apartado, la medida de "cuanto esta olvidando" quedaria falseada.
        corte = int(0.9 * len(datos_pre))
        pre_train, pre_val = datos_pre[:corte], datos_pre[corte:]
        print(f"Corpus original: {len(pre_train):,} tokens para mezclar "
              f"({args.mezcla_pre:.0%} de los lotes) y {len(pre_val):,} apartados "
              f"para medir el olvido")
    elif args.mezcla_pre > 0:
        print("Aviso: --mezcla_pre pedido pero sin --tokens_pre; se entrena solo con instrucciones.")

    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda"))

    con_dec = [q for q in model.parameters() if q.requires_grad and q.dim() >= 2]
    sin_dec = [q for q in model.parameters() if q.requires_grad and q.dim() < 2]
    grupos = [{"params": con_dec, "weight_decay": args.weight_decay},
              {"params": sin_dec, "weight_decay": 0.0}]
    try:
        optimizer = torch.optim.AdamW(grupos, lr=args.lr, fused=(device == "cuda"))
    except (RuntimeError, TypeError) as e:
        print(f"Aviso: sin optimizador fusionado ({e}).")
        optimizer = torch.optim.AdamW(grupos, lr=args.lr)

    warmup = max(1, int(0.03 * args.steps))

    def lr_lambda(paso):
        if paso < warmup:
            return paso / warmup
        avance = (paso - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(avance, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.out_dir, exist_ok=True)
    ruta_parcial = os.path.join(args.out_dir, "atlas_lumerak_parcial.pt")
    ruta_mejor = os.path.join(args.out_dir, "atlas_lumerak_mejor.pt")
    ruta_final = os.path.join(args.out_dir, "atlas_lumerak.pt")

    paso_inicial, mejor_val = 0, float("inf")
    if args.continuar and os.path.exists(ruta_parcial):
        prev = torch.load(ruta_parcial, map_location="cpu")
        model.load_state_dict(quitar_prefijo_compile(prev["model_state_dict"]))
        model.to(device)
        info = prev["entrenamiento"]
        paso_inicial = info["paso"] + 1
        args.steps = info["steps_totales"]
        mejor_val = info.get("mejor_val", float("inf"))
        if "optimizer_state_dict" in prev:
            optimizer.load_state_dict(prev["optimizer_state_dict"])
        with warnings.catch_warnings():
            # Adelantar la curva de tasa de aprendizaje sin optimizar dispara
            # un aviso de PyTorch sobre el orden de las llamadas. Aca es
            # justamente lo que queremos hacer, asi que se silencia.
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(paso_inicial):
                scheduler.step()
        print(f"Retomando en el paso {paso_inicial:,} de {args.steps:,} "
              f"(mejor validacion hasta ahora: {mejor_val:.4f})")

    modelo_paso = model
    if device == "cuda":
        compilado = torch.compile(model)
        try:
            xb, yb = datos.lote(idx_train[:args.batch_size], device)
            with autocast_ctx:
                compilado(xb, yb)
            modelo_paso = compilado
            print("torch.compile activado.")
        except Exception as e:
            print(f"Aviso: sin torch.compile ({type(e).__name__}: {e}).")

    tok.save(os.path.join(args.out_dir, "atlas_bpe.json"))
    info_tok = {"tipo": "bpe", "archivo": "atlas_bpe.json"}

    def guardar(ruta, paso=None, val=None):
        estado = {"model_state_dict": model.state_dict(), "config": config,
                  "tokenizador": info_tok}
        if paso is not None:
            estado["entrenamiento"] = {"paso": paso, "steps_totales": args.steps,
                                       "lr": args.lr, "mejor_val": mejor_val}
            estado["optimizer_state_dict"] = optimizer.state_dict()
        if val is not None:
            estado["val_loss"] = val
        torch.save(estado, ruta + ".tmp")
        os.replace(ruta + ".tmp", ruta)

    print(f"\nEntrenando {args.steps:,} pasos, lote {args.batch_size}, lr {args.lr}\n")
    t0 = time.time()
    rng = np.random.default_rng(123)
    # Promedio de los lotes de conversacion desde la ultima evaluacion. Un
    # solo lote suelto salta demasiado de un paso al otro como para saber si
    # de verdad esta mejorando, y ademas uno de cada diez es de Wikipedia, o
    # sea de otra escala: mezclarlos en el mismo numero seria enganoso.
    suma_sft, n_sft = 0.0, 0
    for paso in range(paso_inicial, args.steps):
        usar_pre = pre_train is not None and rng.random() < args.mezcla_pre
        if usar_pre:
            xb, yb = lote_pre(pre_train, block_size, args.batch_size, device)
        else:
            sel = idx_train[torch.randint(len(idx_train), (args.batch_size,))]
            xb, yb = datos.lote(sel, device)

        with autocast_ctx:
            _, loss = modelo_paso(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if not usar_pre:
            suma_sft += loss.item()
            n_sft += 1

        if paso % args.eval_interval == 0 or paso == args.steps - 1:
            val = evaluar(modelo_paso, datos, idx_val, args.batch_size, device, autocast_ctx)
            marca = ""
            if val < mejor_val:
                mejor_val = val
                guardar(ruta_mejor, val=val)
                marca = "  <-- mejor, guardado"
            entrenamiento = f"{suma_sft / n_sft:.4f}" if n_sft else "  -   "
            suma_sft, n_sft = 0.0, 0
            olvido = ""
            if pre_val is not None:
                w = evaluar_olvido(modelo_paso, pre_val, block_size, args.batch_size,
                                   device, autocast_ctx)
                olvido = f" | wikipedia {w:.4f}"
            print(f"paso {paso:5d}/{args.steps} | entrenamiento {entrenamiento} | "
                  f"respuestas {val:.4f}{olvido} | "
                  f"{time.time() - t0:.0f}s{marca}", flush=True)

        if args.save_interval > 0 and paso > 0 and paso % args.save_interval == 0:
            guardar(ruta_parcial, paso=paso)

    guardar(ruta_final)
    if os.path.exists(ruta_parcial):
        os.remove(ruta_parcial)
    print(f"\nUltimo paso guardado en:  {ruta_final}")
    print(f"MEJOR modelo guardado en: {ruta_mejor}  (validacion {mejor_val:.4f})")
    print("Usa el MEJOR para chatear; el ultimo casi siempre esta mas sobreajustado.")


if __name__ == "__main__":
    main()
