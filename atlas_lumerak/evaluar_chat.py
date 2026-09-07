"""
PASO 13: el examen automatico. La vara de medir.

Por que existe: hasta ahora juzgabamos a Atlas con diez preguntas y el ojo.
Eso ya nos engano dos veces. Una vez saque una conclusion firme de cinco
respuestas y estaba equivocada; otra vez casi propongo dos horas de
reentrenamiento para arreglar un bug de cinco lineas.

Un modelo de lenguaje responde distinto cada vez, asi que una respuesta
suelta no dice nada. Este examen hace muchas preguntas, varias veces cada
una, y cuenta. Nada de opiniones: numeros que se pueden comparar entre dos
modelos, entre dos ajustes o entre dos meses.

Que mide, y por que cada cosa:

  IDENTIDAD    ¿dice como se llama y quien lo creo? Es lo unico que no
               puede haber aprendido de ningun texto ajeno.
  ESPANOL      ¿responde en espanol? Suena absurdo hasta que un bug de
               muestreo lo empuja al ingles roto, que es lo que paso.
  OBEDIENCIA   pides una lista de 4 cosas, ¿da 4? Mide si sigue la
               instruccion o solo imita su forma.
  RELEVANCIA   ¿la respuesta habla de lo que se pregunto? Era el fallo
               original: contestaba sobre villancicos cuando le
               preguntabas su nombre.
  REPETICION   ¿se traba repitiendo? Defecto tipico de los modelos chicos.
  PARADA       ¿termina la frase o se corta a media palabra al agotar los
               tokens? Mide si sabe cuando callarse.

El resultado se guarda en un JSON con fecha y modelo, para poder mirar la
serie completa dentro de seis meses y ver la curva.
"""

import argparse
import json
import os
import re
import time
import unicodedata

import torch

from checkpoint_utils import cargar_modelo
from inferencia import recortar_respuesta

# ---------------------------------------------------------------- preguntas
IDENTIDAD = [
    ("¿Cómo te llamas?", "nombre"), ("¿Cuál es tu nombre?", "nombre"),
    ("Dime tu nombre", "nombre"), ("como te llamas", "nombre"),
    ("¿Quién eres?", "nombre"), ("Preséntate", "nombre"),
    ("¿Quién te creó?", "creador"), ("¿Quién te hizo?", "creador"),
    ("quien te creo", "creador"), ("¿De dónde vienes?", "creador"),
]
# (instruccion, cuantos elementos se piden)
LISTAS = [
    ("Escribe una lista de 3 frutas", 3),
    ("Escribe una lista de 4 colores", 4),
    ("Dame 3 ideas para estudiar mejor", 3),
    ("Nombra 5 animales", 5),
    ("Dame 2 consejos para dormir mejor", 2),
]
# (pregunta, palabras que una respuesta relevante deberia mencionar)
RELEVANCIA = [
    ("¿Cuál es la capital de Francia?", ["parís", "paris", "francia"]),
    ("¿Qué es la fotosíntesis?", ["planta", "luz", "energía", "energia", "fotosíntesis", "fotosintesis"]),
    ("¿Para qué sirve el agua?", ["agua", "beber", "vida", "hidrat"]),
    ("¿Qué es un perro?", ["perro", "animal", "mamífero", "mamifero", "can"]),
    ("Explícame qué es el sol", ["sol", "estrella", "luz", "calor"]),
    ("¿Qué es la lluvia?", ["lluvia", "agua", "nube", "precipit"]),
    ("¿Cómo se llama el planeta en el que vivimos?", ["tierra"]),
    ("¿Qué es un libro?", ["libro", "leer", "página", "pagina", "texto", "escrit"]),
]
CONVERSACION = [
    "Hola", "Hola, buenos días", "¿Cómo estás?", "Gracias", "Adiós",
]

# Palabras muy frecuentes de cada idioma, para saber en cual esta escrita
# una respuesta sin depender de ninguna libreria externa.
PAL_ES = set("de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este sí porque esta cuando muy sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros".split())
PAL_EN = set("the of and to in is that it for with was are you your this be have not on as at by from or an we they he she his her their there but what all can will one would about which when who been has had do does did if my me so no out up".split())


def sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower())
                   if unicodedata.category(c) != "Mn")


def es_espanol(texto: str) -> bool | None:
    """None si la respuesta es demasiado corta para decidir."""
    pal = re.findall(r"[a-záéíóúñü]+", texto.lower())
    if len(pal) < 6:
        return None
    ne = sum(p in PAL_EN for p in pal)
    nes = sum(p in PAL_ES for p in pal)
    if ne == 0 and nes == 0:
        return None
    return nes >= ne


def contar_elementos(texto: str) -> int:
    """Cuenta los elementos de una lista, numerada o con guiones."""
    n = len(re.findall(r"^\s*\d+[\.\)]\s+\S", texto, re.M))
    return n or len(re.findall(r"^\s*[-*•]\s+\S", texto, re.M))


def trigramas_repetidos(texto: str) -> float:
    pal = texto.lower().split()
    if len(pal) < 6:
        return 0.0
    tri = [tuple(pal[i:i + 3]) for i in range(len(pal) - 2)]
    return 1 - len(set(tri)) / len(tri)


def termina_limpio(texto: str) -> bool:
    return bool(texto) and texto.rstrip()[-1:] in ".!?\"')…:"


# ---------------------------------------------------------------- ejecucion
def responder(model, tok, block_size, device, pregunta, args, semilla):
    torch.manual_seed(semilla)
    ids = tok.encode(f"Usuario: {pregunta}\nAtlas:")[-block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    salida = model.generate(
        idx, max_new_tokens=args.length, temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p < 1.0 else None,
        repetition_penalty=args.repeticion,
    )[0].tolist()
    crudo = tok.decode(salida[len(ids):])
    return recortar_respuesta(crudo), crudo


def main():
    p = argparse.ArgumentParser(description="Examen automatico de Atlas Lumerak.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--muestras", type=int, default=5,
                   help="Cuantas veces se hace cada pregunta. Mas = medida mas estable.")
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repeticion", type=float, default=1.15)
    p.add_argument("--historial", default="atlas_lumerak/data/examenes.jsonl")
    p.add_argument("--ejemplos", type=int, default=3, help="Cuantas respuestas mostrar al final.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, config = cargar_modelo(args.checkpoint, device, args.vocab)
    bs = config["block_size"]
    n_par = sum(x.numel() for x in model.parameters())
    print(f"Modelo: {args.checkpoint}")
    print(f"  {n_par:,} parametros | temperatura {args.temperature}, top_k {args.top_k}, "
          f"top_p {args.top_p}, repeticion {args.repeticion}")
    total_preg = (len(IDENTIDAD) + len(LISTAS) + len(RELEVANCIA) + len(CONVERSACION)) * args.muestras
    print(f"  {total_preg} respuestas a generar...\n", flush=True)

    t0 = time.time()
    todas, muestrario = [], []

    def pedir(pregunta, k):
        """Devuelve (respuesta recortada, respuesta sin recortar)."""
        return responder(model, tok, bs, device, pregunta, args, 1000 + k * 7)

    # --- identidad ---
    aciertos_nombre = total_nombre = aciertos_creador = total_creador = 0
    for pregunta, tipo in IDENTIDAD:
        for k in range(args.muestras):
            r, _ = pedir(pregunta, k); todas.append(r)
            plano = sin_tildes(r)
            if tipo == "nombre":
                total_nombre += 1
                aciertos_nombre += "atlas lumerak" in plano
            else:
                total_creador += 1
                aciertos_creador += "sebastian" in plano
        muestrario.append((pregunta, r))

    # --- obediencia de listas ---
    # Se cuenta sobre la respuesta recortada Y sobre la entera. Si las dos
    # cifras difieren mucho, el fallo no es de Atlas contando: es del
    # recorte, que corta la lista en la primera linea en blanco. Distinguir
    # "no sabe contar" de "lo mide mal mi codigo" es justo el motivo de que
    # este examen exista.
    ok_listas = ok_listas_crudo = total_listas = 0
    detalle_listas = []
    ejemplo_lista_crudo = ""
    for pregunta, esperados in LISTAS:
        for k in range(args.muestras):
            r, crudo = pedir(pregunta, k); todas.append(r)
            n_rec, n_crudo = contar_elementos(r), contar_elementos(crudo)
            total_listas += 1
            ok_listas += n_rec == esperados
            ok_listas_crudo += n_crudo == esperados
            detalle_listas.append({"pedidos": esperados, "dados": n_crudo})
            if n_crudo > n_rec and not ejemplo_lista_crudo:
                ejemplo_lista_crudo = crudo   # un caso donde el recorte perdio elementos
        muestrario.append((pregunta, r))

    # --- relevancia ---
    ok_rel = total_rel = 0
    for pregunta, claves in RELEVANCIA:
        for k in range(args.muestras):
            r, _ = pedir(pregunta, k); todas.append(r)
            total_rel += 1
            ok_rel += any(c in sin_tildes(r) for c in map(sin_tildes, claves))
        muestrario.append((pregunta, r))

    # --- conversacion (solo alimenta las metricas globales) ---
    for pregunta in CONVERSACION:
        for k in range(args.muestras):
            r, _ = pedir(pregunta, k); todas.append(r)
        muestrario.append((pregunta, r))

    # --- metricas sobre TODAS las respuestas ---
    juicios = [es_espanol(r) for r in todas]
    decidibles = [j for j in juicios if j is not None]
    # Las respuestas de menos de 6 palabras no se pueden clasificar por
    # idioma con este metodo. Se informa cuantas SI se pudieron juzgar: un
    # "100% espanol" calculado sobre tres respuestas no significa nada, y
    # sin el tamano de la muestra el porcentaje engana.
    espanol = sum(decidibles) / len(decidibles) * 100 if decidibles else float("nan")
    repes = sum(trigramas_repetidos(r) for r in todas) / len(todas) * 100
    limpias = sum(termina_limpio(r) for r in todas) / len(todas) * 100
    vacias = sum(not r.strip() for r in todas) / len(todas) * 100
    largo = sum(len(r.split()) for r in todas) / len(todas)

    res = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "parametros": n_par,
        "muestreo": {"temperature": args.temperature, "top_k": args.top_k,
                     "top_p": args.top_p, "repeticion": args.repeticion,
                     "length": args.length, "muestras": args.muestras},
        "identidad_nombre": aciertos_nombre / total_nombre * 100,
        "identidad_creador": aciertos_creador / total_creador * 100,
        "espanol": espanol,
        "espanol_evaluables": len(decidibles),
        # La nota oficial se toma sobre la respuesta ENTERA, no sobre la
        # recortada. El examen mide a Atlas; el recorte es una decision de
        # presentacion del chat y no debe contaminar la medida del modelo.
        # Se conserva la otra cifra aparte, porque es lo que de verdad ve
        # el usuario y su diferencia mide la calidad del recorte.
        "obediencia_listas": ok_listas_crudo / total_listas * 100,
        "obediencia_listas_tras_recorte": ok_listas / total_listas * 100,
        "detalle_listas": detalle_listas,
        "relevancia": ok_rel / total_rel * 100,
        "repeticion": repes,
        "parada_limpia": limpias,
        "respuestas_vacias": vacias,
        "palabras_por_respuesta": largo,
        "n_respuestas": len(todas),
    }

    print("=" * 62)
    print(f"{'EXAMEN DE ATLAS LUMERAK':^62}")
    print("=" * 62)
    filas = [
        ("Dice su nombre", res["identidad_nombre"], "mas es mejor"),
        ("Dice quien lo creo", res["identidad_creador"], "mas es mejor"),
        ("Responde en espanol", res["espanol"],
         f"mas es mejor (sobre {len(decidibles)} de {len(todas)} juzgables)"),
        ("Obedece el numero pedido", res["obediencia_listas"], "mas es mejor"),
        ("Habla del tema preguntado", res["relevancia"], "mas es mejor"),
        ("Termina la frase", res["parada_limpia"], "mas es mejor"),
        ("Se repite", res["repeticion"], "MENOS es mejor"),
        ("Respuestas vacias", res["respuestas_vacias"], "MENOS es mejor"),
    ]
    for nombre, valor, sentido in filas:
        barra = "#" * round(valor / 4)
        print(f"  {nombre:26s} {valor:5.1f}%  {barra:<25s} {sentido}")
    # Diagnostico de las listas: cuantos elementos pidio y cuantos dio.
    from collections import Counter
    print(f"\n  Listas: {res['obediencia_listas']:.0f}% acierta (lo que genera Atlas); "
          f"{res['obediencia_listas_tras_recorte']:.0f}% sobrevive al recorte del chat")
    reparto = Counter((d["pedidos"], d["dados"]) for d in detalle_listas)
    for (ped, dad), n in sorted(reparto.items()):
        marca = " <-- correcto" if ped == dad else ""
        print(f"    pidio {ped}, dio {dad:2d}  ({n} veces){marca}")

    print(f"\n  Longitud media: {largo:.0f} palabras por respuesta")
    print(f"  {len(todas)} respuestas en {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.historial) or ".", exist_ok=True)
    with open(args.historial, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"\n  Guardado en {args.historial} (una linea por examen, para ver la curva)")

    if ejemplo_lista_crudo:
        print("\n--- una respuesta de lista, tal cual la escribe Atlas ---")
        print("   ", repr(ejemplo_lista_crudo[:400]))

    if args.ejemplos:
        print("\n--- algunas respuestas, para mirarlas con los ojos ---")
        paso = max(1, len(muestrario) // args.ejemplos)
        for pregunta, r in muestrario[::paso][:args.ejemplos]:
            print(f"\n  P: {pregunta}\n  R: {(r or '(vacia)')[:300]}")


if __name__ == "__main__":
    main()
