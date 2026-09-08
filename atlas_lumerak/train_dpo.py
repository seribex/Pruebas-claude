"""
PASO 15: Atlas aprende de sus propios errores.

Todo el entrenamiento anterior le ensenaba lo mismo: "asi se responde".
Nunca le dijimos "esto que TU dijiste, no". Y hay una diferencia enorme
entre darle a alguien la solucion y senalarle donde se equivoco: lo primero
le muestra un camino bueno entre muchos, lo segundo le cierra el malo que
estaba tomando.

Esto entrena con pares: la misma pregunta, la respuesta buena y la
equivocacion concreta de Atlas. El objetivo no es "produce la buena" sino
"prefiere la buena ANTES que esa". La tecnica se llama DPO (optimizacion
directa por preferencias) y es la que usan los modelos actuales despues de
su entrenamiento normal.

Como funciona, sin formulas: se guarda una copia congelada de Atlas tal
como esta hoy, que sirve de referencia. Se le pide a las dos versiones -- la
que aprende y la congelada -- que digan cuanto de probable les parece cada
respuesta. Y se empuja a la que aprende para que suba la buena y baje la
mala MAS DE LO QUE LO HACIA ANTES. La copia congelada es lo que impide que
se desmadre: sin ella, la forma mas facil de "preferir A sobre B" es
volverse absurdo, y el modelo se destruiria a si mismo.

Requiere pocos pasos y tasa de aprendizaje baja: no esta aprendiendo nada
nuevo, esta corrigiendo inclinaciones que ya tiene.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizer, FIN_TURNO
from checkpoint_utils import quitar_prefijo_compile
from model import TransformerLanguageModel

IGNORAR = -100


def cargar_pares(rutas, tok, block_size):
    """Convierte cada correccion en (entrada, objetivo) para la buena y la mala."""
    id_fin = tok.especiales.get(FIN_TURNO)
    pares, saltados = [], 0
    for ruta in rutas:
        if not os.path.exists(ruta):
            print(f"  (no existe: {ruta})")
            continue
        for linea in open(ruta, encoding="utf-8"):
            d = json.loads(linea)
            buena, mala = d["buena"].strip(), d["mala"].strip()
            # Una respuesta vacia SI es algo que corregir: el par pasa a ser
            # "di esto" contra "no digas nada", que es una preferencia
            # perfectamente valida y ademas un fallo real que hemos visto.
            if not buena or buena == mala:
                saltados += 1
                continue
            ip = tok.encode(f"Usuario: {d['pregunta']}\nAtlas:")
            secuencias = []
            for texto in (buena, mala):
                ir = tok.encode(f" {texto}") + ([id_fin] if id_fin is not None else [])
                ids = (ip + ir)[:block_size]
                if len(ids) <= len(ip):
                    break
                objetivo = list(ids)
                objetivo[:len(ip)] = [IGNORAR] * len(ip)
                secuencias.append((ids, objetivo))
            if len(secuencias) == 2:
                pares.append(secuencias)
            else:
                saltados += 1
    return pares, saltados


def apilar(secuencias, device):
    """Junta secuencias de distinto largo rellenando por la derecha.

    Rellenar por la DERECHA es seguro aunque el modelo no sepa de relleno:
    la atencion es causal, asi que cada posicion solo mira hacia atras y el
    relleno del final no puede contaminar nada anterior. Y las posiciones de
    relleno se marcan como ignoradas, o sea que no aportan al resultado."""
    largo = max(len(ids) for ids, _ in secuencias)
    x = torch.zeros(len(secuencias), largo, dtype=torch.long)
    y = torch.full((len(secuencias), largo), IGNORAR, dtype=torch.long)
    for i, (ids, objetivo) in enumerate(secuencias):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(objetivo)] = torch.tensor(objetivo)
    return x.to(device), y.to(device)


def log_prob_respuesta(modelo, x, y, autocast_ctx):
    """Cuanto de probable le parece al modelo la respuesta (solo la respuesta)."""
    with autocast_ctx:
        logits, _ = modelo(x)
    logits = logits[:, :-1].float()
    objetivo = y[:, 1:]
    valido = objetivo != IGNORAR
    seguro = objetivo.masked_fill(~valido, 0)
    lp = torch.log_softmax(logits, dim=-1).gather(2, seguro.unsqueeze(-1)).squeeze(-1)
    return (lp * valido).sum(dim=1)


def main():
    p = argparse.ArgumentParser(description="Atlas aprende de sus errores (DPO).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bpe", default=None)
    p.add_argument("--correcciones", nargs="+",
                   default=["atlas_lumerak/data/correcciones.jsonl",
                            "atlas_lumerak/data/correcciones_tuyas.jsonl"])
    p.add_argument("--out_dir", default="atlas_lumerak/checkpoints_dpo")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epocas", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="Muy baja a proposito: no esta aprendiendo nada nuevo, solo corrigiendo "
                        "inclinaciones que ya tiene. Con una tasa alta se desarma.")
    p.add_argument("--beta", type=float, default=0.1,
                   help="Cuanto se le permite alejarse de la copia congelada. Mas alto = mas "
                        "conservador.")
    p.add_argument("--eval_interval", type=int, default=25)
    p.add_argument("--val_frac", type=float, default=0.1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    torch.manual_seed(42)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    estado = quitar_prefijo_compile(ckpt["model_state_dict"])
    ruta_bpe = args.bpe or os.path.join(os.path.dirname(args.checkpoint),
                                        ckpt.get("tokenizador", {}).get("archivo", "atlas_bpe.json"))
    tok = BPETokenizer.load(ruta_bpe)

    pares, saltados = cargar_pares(args.correcciones, tok, config["block_size"])
    if not pares:
        print("No hay correcciones con las que entrenar. Corre practicar.py primero.")
        return
    print(f"Correcciones cargadas: {len(pares):,} (descartadas {saltados:,})")

    # La que aprende y la copia congelada que sirve de referencia.
    politica = TransformerLanguageModel(**config).to(device)
    politica.load_state_dict(estado)
    referencia = TransformerLanguageModel(**config).to(device)
    referencia.load_state_dict(estado)
    referencia.eval()
    for q in referencia.parameters():
        q.requires_grad_(False)
    print(f"Modelo: {sum(q.numel() for q in politica.parameters()):,} parametros "
          f"(+ una copia congelada de referencia)")

    g = torch.Generator().manual_seed(7)
    orden = torch.randperm(len(pares), generator=g).tolist()
    # La validacion nunca puede quedarse con todo: con pocas correcciones,
    # se reserva como mucho la mitad y siempre queda algo para entrenar.
    n_val = min(max(1, int(args.val_frac * len(pares))), max(1, len(pares) // 2))
    idx_val, idx_train = orden[:n_val], orden[n_val:]
    if not idx_train:
        print("Hacen falta al menos 2 correcciones para entrenar.")
        return
    pasos = max(1, int(args.epocas * len(idx_train) / args.batch_size))
    print(f"Entrenamiento: {len(idx_train):,} pares | validacion: {len(idx_val):,} | "
          f"{pasos:,} pasos\n")

    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16,
                                  enabled=(device == "cuda"))
    con = [q for q in politica.parameters() if q.dim() >= 2]
    sin = [q for q in politica.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": con, "weight_decay": 0.0},
                             {"params": sin, "weight_decay": 0.0}], lr=args.lr)

    def perdida(indices):
        secuencias = []
        for i in indices:
            secuencias.append(pares[i][0])   # buena
        for i in indices:
            secuencias.append(pares[i][1])   # mala
        x, y = apilar(secuencias, device)
        n = len(indices)
        lp = log_prob_respuesta(politica, x, y, autocast_ctx)
        with torch.no_grad():
            lp_ref = log_prob_respuesta(referencia, x, y, autocast_ctx)
        margen = (lp[:n] - lp_ref[:n]) - (lp[n:] - lp_ref[n:])
        return -F.logsigmoid(args.beta * margen).mean(), (margen > 0).float().mean()

    @torch.no_grad()
    def evaluar():
        politica.eval()
        total = acierto = 0.0
        lotes = max(1, len(idx_val) // args.batch_size)
        for k in range(lotes):
            sel = idx_val[k * args.batch_size:(k + 1) * args.batch_size]
            if not sel:
                break
            l, a = perdida(sel)
            total += l.item(); acierto += a.item()
        politica.train()
        return total / lotes, acierto / lotes

    os.makedirs(args.out_dir, exist_ok=True)
    tok.save(os.path.join(args.out_dir, "atlas_bpe.json"))
    print("El primer valor de perdida debe ser 0.6931 (= log 2): al empezar, la que")
    print("aprende y la congelada son identicas y no prefiere ninguna. Si no sale eso,")
    print("algo esta mal conectado.\n")

    t0, mejor = time.time(), float("inf")
    for paso in range(pasos):
        sel = [idx_train[(paso * args.batch_size + k) % len(idx_train)]
               for k in range(args.batch_size)]
        l, a = perdida(sel)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(politica.parameters(), 1.0)
        opt.step()

        if paso % args.eval_interval == 0 or paso == pasos - 1:
            vl, va = evaluar()
            marca = ""
            if vl < mejor:
                mejor = vl
                torch.save({"model_state_dict": politica.state_dict(), "config": config,
                            "tokenizador": {"tipo": "bpe", "archivo": "atlas_bpe.json"},
                            "dpo_val": vl},
                           os.path.join(args.out_dir, "atlas_lumerak_mejor.pt"))
                marca = "  <-- mejor, guardado"
            print(f"paso {paso:5d}/{pasos} | perdida {l.item():.4f} | "
                  f"validacion {vl:.4f} | prefiere la buena {va*100:5.1f}% | "
                  f"{time.time()-t0:.0f}s{marca}", flush=True)

    print(f"\nGuardado en {args.out_dir}/atlas_lumerak_mejor.pt")
    print("Compara con evaluar_chat.py antes de darlo por bueno: DPO puede mejorar")
    print("lo que se le corrigio y empeorar el resto.")


if __name__ == "__main__":
    main()
