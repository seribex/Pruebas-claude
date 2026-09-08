"""
PASO 14: Atlas practica, se equivoca, y la equivocacion queda registrada.

La idea es la que pidio Sebastian: que Atlas converse de verdad, que alguien
lo corrija y que aprenda de sus propios errores. La parte no obvia es QUIEN
corrige.

Lo intuitivo seria ponerlo a hablar con otra IA por internet. Pero eso
cuesta dinero por cada correccion, es lento, y es justo la dependencia que
no queremos. Y no hace falta: el corpus YA es un maestro con un millon de
lecciones preparadas. Cada pregunta viene con su respuesta buena al lado.

Asi que aqui Atlas responde a preguntas cuya respuesta correcta ya
conocemos, un juez automatico comprueba si metio la pata, y cuando la mete
se guarda el par completo:

    pregunta   ->  lo que Atlas dijo (mal)  |  lo que deberia haber dicho

Eso es material de entrenamiento de una clase distinta al que teniamos. El
entrenamiento normal solo le muestra la respuesta correcta; esto le muestra
ademas SU error concreto, para empujarlo lejos de el. Es la diferencia entre
darle a alguien la solucion y senalarle donde se equivoco.

El juez solo comprueba cosas OBJETIVAS -- si obedecio el numero pedido, si
contesto en espanol, si dijo su nombre cuando se lo preguntaron, si se
quedo en bucle. No pretende juzgar si una respuesta es "buena": eso no se
puede automatizar honestamente, y fingir que si llenaria el archivo de
correcciones falsas.
"""

import argparse
import json
import os
import time

import torch

import identidad
import listas
import prepare_sft
from bpe_tokenizer import FIN_TURNO
from checkpoint_utils import cargar_modelo
from evaluar_chat import contar_elementos, sin_tildes, trigramas_repetidos
from inferencia import id_fin_de_turno, recortar_respuesta
from text_cleaning import es_espanol

# (pregunta, respuesta correcta, tipo, dato que necesita el juez)
Ejercicio = tuple[str, str, str, object]


def juzgar(tipo, dato, respuesta: str) -> str | None:
    """Devuelve el motivo del fallo, o None si la respuesta pasa.

    Solo comprueba defectos objetivos. Que una respuesta sea acertada o no
    es algo que no se puede juzgar automaticamente sin mentir."""
    if not respuesta.strip():
        return "respuesta vacia"
    if es_espanol(respuesta) is False:
        return "responde en otro idioma"
    if trigramas_repetidos(respuesta) > 0.35:
        return "se queda en bucle repitiendo"

    if tipo == "identidad" and dato:
        if not any(clave in sin_tildes(respuesta) for clave in dato):
            return f"no menciona lo que debia ({', '.join(dato)})"
    if tipo == "lista":
        dados = contar_elementos(respuesta)
        if dados != dato:
            return f"se le pidieron {dato} elementos y dio {dados}"
    return None


def reunir_ejercicios(args) -> list[Ejercicio]:
    ejercicios: list[Ejercicio] = []

    # Identidad: se sabe exactamente que tiene que aparecer en la respuesta.
    claves = {"nombre": ["atlas lumerak"], "creador": [sin_tildes(identidad.CREADOR)],
              "no_soy_mi_creador": ["atlas lumerak"]}
    for intencion, datos in identidad.INTENCIONES.items():
        for i, pregunta in enumerate(datos["preguntas"]):
            respuesta = datos["respuestas"][i % len(datos["respuestas"])]
            ejercicios.append((pregunta, respuesta, "identidad", claves.get(intencion)))

    # Listas: el numero pedido se conoce por construccion.
    for conv in listas.ejemplos(args.pasadas_listas):
        pregunta, buena = conv[0]
        ejercicios.append((pregunta, buena, "lista", contar_elementos(buena)))

    # Corpus general: no se puede juzgar el acierto, pero si los defectos.
    if not args.sin_corpus:
        for cargar in (prepare_sft.cargar_oasst2, prepare_sft.cargar_dolly):
            for conv in cargar(args.tmp):
                pregunta, buena = conv[-1]
                if len(buena) <= args.max_chars:
                    ejercicios.append((pregunta, buena, "general", None))
    return ejercicios


def main():
    p = argparse.ArgumentParser(description="Atlas practica y sus errores quedan registrados.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--n", type=int, default=5000, help="Cuantos ejercicios hacer.")
    p.add_argument("--length", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Mas alta que en el chat a proposito: aqui queremos ver la variedad "
                        "de errores que PUEDE cometer, no su mejor respuesta.")
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repeticion", type=float, default=1.15)
    p.add_argument("--salida", default="atlas_lumerak/data/correcciones.jsonl")
    p.add_argument("--tmp", default="atlas_lumerak/data/_tmp_sft")
    p.add_argument("--pasadas_listas", type=int, default=3)
    p.add_argument("--max_chars", type=int, default=600)
    p.add_argument("--sin_corpus", action="store_true")
    p.add_argument("--semilla", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, config = cargar_modelo(args.checkpoint, device, args.vocab)
    bs, id_fin = config["block_size"], id_fin_de_turno(tok)
    print(f"Modelo: {args.checkpoint}")

    ejercicios = reunir_ejercicios(args)
    g = torch.Generator().manual_seed(args.semilla)
    orden = torch.randperm(len(ejercicios), generator=g)[:args.n]
    print(f"Ejercicios disponibles: {len(ejercicios):,} | se practicaran {len(orden):,}\n")

    correcciones, aciertos = [], 0
    t0 = time.time()
    for k, idx_ej in enumerate(orden.tolist(), 1):
        pregunta, buena, tipo, dato = ejercicios[idx_ej]
        torch.manual_seed(args.semilla + k)
        ids = tok.encode(f"Usuario: {pregunta}\nAtlas:")[-bs:]
        salida = model.generate(
            torch.tensor([ids], dtype=torch.long, device=device),
            max_new_tokens=args.length, temperature=args.temperature,
            top_k=args.top_k or None, top_p=args.top_p if args.top_p < 1 else None,
            repetition_penalty=args.repeticion, stop_id=id_fin,
        )[0].tolist()
        dicha = recortar_respuesta(tok.decode(salida[len(ids):]))

        motivo = juzgar(tipo, dato, dicha)
        if motivo is None:
            aciertos += 1
        else:
            correcciones.append({"pregunta": pregunta, "mala": dicha, "buena": buena,
                                 "motivo": motivo, "tipo": tipo})

        if k % 250 == 0:
            print(f"  {k:6,}/{len(orden):,} | aciertos {aciertos/k*100:5.1f}% | "
                  f"correcciones {len(correcciones):,} | {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        for c in correcciones:
            f.write(json.dumps({**c, "checkpoint": args.checkpoint,
                                "fecha": time.strftime("%Y-%m-%d %H:%M:%S")},
                               ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"\n{'='*62}\n{'RESULTADO DE LA PRACTICA':^62}\n{'='*62}")
    print(f"  ejercicios: {len(orden):,} | acertados: {aciertos:,} "
          f"({aciertos/len(orden)*100:.1f}%)")
    print(f"  CORRECCIONES REGISTRADAS: {len(correcciones):,}\n")
    for motivo, n in Counter(c["motivo"].split(" y dio")[0] for c in correcciones).most_common(8):
        print(f"    {n:6,}  {motivo}")
    print(f"\n  Guardadas en {args.salida}")
    print(f"  Cada linea es una leccion: la pregunta, lo que Atlas dijo mal,")
    print(f"  y lo que deberia haber dicho.")


if __name__ == "__main__":
    main()
