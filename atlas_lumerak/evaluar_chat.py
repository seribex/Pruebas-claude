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
import math
import os
import re
import time
import unicodedata

import torch

from checkpoint_utils import cargar_modelo
from text_cleaning import es_espanol
from inferencia import id_fin_de_turno, recortar_respuesta

# ---------------------------------------------------------------- preguntas
IDENTIDAD = [
    ("¿Cómo te llamas?", "nombre"), ("¿Cuál es tu nombre?", "nombre"),
    ("Dime tu nombre", "nombre"), ("como te llamas", "nombre"),
    ("¿Quién eres?", "nombre"), ("Preséntate", "nombre"),
    ("¿Quién te creó?", "creador"), ("¿Quién te hizo?", "creador"),
    ("quien te creo", "creador"), ("¿De dónde vienes?", "creador"),
    ("¿Quién es tu creador?", "creador"), ("¿Quién te programó?", "creador"),
    ("Dime el nombre de tu creador", "creador"), ("quien esta detras de ti", "creador"),
]
# (instruccion, cuantos elementos se piden)
LISTAS = [
    ("Escribe una lista de 3 frutas", 3),
    ("Escribe una lista de 4 colores", 4),
    ("Dame 3 ideas para estudiar mejor", 3),
    ("Nombra 5 animales", 5),
    ("Dame 2 consejos para dormir mejor", 2),
    ("Escribe una lista de 3 países", 3),
    ("Dame 4 ejemplos de deportes", 4),
    ("Nombra 2 instrumentos musicales", 2),
    ("Escribe 5 profesiones", 5),
    ("Dame 3 nombres de ríos", 3),
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

# Conversaciones enteras, con el contexto acumulandose turno a turno igual
# que en chat.py. Existen porque el examen tenia un punto ciego: media 97.8%
# de espanol haciendo preguntas sueltas, y en conversacion real el ingles
# aparecia en 3 de cada 8 turnos. Lo que se usa es la conversacion, no la
# pregunta aislada, asi que hay que medir eso.
#
# Se escriben como escribe la gente en un chat: sin tildes, sin signos de
# apertura y en minusculas. Atlas tiene que aguantarlo.
# Cada turno es (pregunta, lo que una respuesta correcta debe contener).
# None = no se comprueba el contenido, solo alimenta el contexto.
#
# Las preguntas de identidad aparecen aqui en los turnos 2 y 3 a proposito:
# el examen las hacia solo con el contexto limpio y daba 82.5%, pero en el
# chat real fallaban. Habia que medir tambien esa situacion.
CONVERSACIONES = [
    [("Hola", None), ("¿Cómo te llamas?", ["atlas lumerak"]),
     ("¿Quién te creó?", ["sebastian"])],
    [("Que es la fotosintesis", None), ("y para que sirve", None),
     ("como te llamas", ["atlas lumerak"])],
    [("Hola, buenos dias", None), ("que es una guerra", None),
     ("quien te creo", ["sebastian"])],
    [("Dame 3 ideas para estudiar mejor", None), ("cual es la mejor", None),
     ("por que", None)],
    [("¿Cuál es la capital de Francia?", None), ("gracias", None),
     ("¿Quién eres?", ["atlas lumerak"])],
]

def primera_frase(texto: str) -> str:
    """Lo que Atlas dice antes de su primer punto o salto de linea.

    Existe para comprobar una sospecha concreta: que acierta al principio de
    la respuesta y se descarrila despues. Si la relevancia de la primera
    frase es claramente mayor que la del texto entero, la conclusion es que
    le estamos pidiendo respuestas mas largas de las que puede sostener.
    """
    # Un punto detras de un numero AL PRINCIPIO DE LINEA es el marcador de un
    # elemento de lista ("1. Manzana"), no el final de una frase. En cambio
    # "El agua hierve a 100." si termina una frase. Los distingue la posicion.
    marcadores = {m.end() - 1 for m in re.finditer(r"(?m)^[ \t]*\d+[.)]", texto)}
    for m in re.finditer(r"[.!?](?=\s|$)|\n", texto):
        if m.start() in marcadores:
            continue
        return texto[:m.end()].strip() or texto.strip()
    return texto.strip()


def intervalo(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza del 95% (metodo de Wilson).

    Un porcentaje sin su intervalo engana: con 25 respuestas, un 24% y un
    12% son indistinguibles -- puro azar del muestreo. Nos paso: interprete
    como mejoras y empeoramientos diferencias que no significaban nada.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = exitos / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    radio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centro - radio) * 100, min(1.0, centro + radio) * 100)


def sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower())
                   if unicodedata.category(c) != "Mn")


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
        stop_id=id_fin_de_turno(tok),
    )[0].tolist()
    crudo = tok.decode(salida[len(ids):])
    return recortar_respuesta(crudo), crudo


def main():
    p = argparse.ArgumentParser(description="Examen automatico de Atlas Lumerak.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--muestras", type=int, default=20,
                   help="Cuantas veces se hace cada pregunta. Con 5 (el valor anterior) los "
                        "intervalos eran de +-15 puntos y no se distinguia una mejora del "
                        "azar. Para detectar un cambio de 10 puntos hacen falta del orden de "
                        "250 respuestas por medida.")
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repeticion", type=float, default=1.15)
    p.add_argument("--historial", default="atlas_lumerak/data/examenes.jsonl")
    p.add_argument("--ejemplos", type=int, default=3, help="Cuantas respuestas mostrar al final.")
    p.add_argument("--errores", type=int, default=4,
                   help="Cuantos ejemplos de cada tipo de error imprimir. Todos se guardan "
                        "ademas en data/errores.jsonl.")
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
    # Cada respuesta equivocada se guarda entera, con lo que se le pidio y
    # por que fallo. Un porcentaje dice CUANTO falla; solo el error concreto
    # dice POR QUE, y sin eso no hay nada que corregir.
    fallos: list[dict] = []

    def anotar(categoria, pregunta, respuesta, ok, esperado=""):
        if not ok:
            fallos.append({"categoria": categoria, "pregunta": pregunta,
                           "respuesta": respuesta, "esperado": esperado})

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
                ok = "atlas lumerak" in plano
                aciertos_nombre += ok
                anotar("no dice su nombre", pregunta, r, ok, "Atlas Lumerak")
            else:
                total_creador += 1
                ok = "sebastian" in plano
                aciertos_creador += ok
                anotar("no dice quien lo creo", pregunta, r, ok, "Sebastian")
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
            anotar("no obedece el numero pedido", pregunta, crudo,
                   n_crudo == esperados, f"{esperados} elementos, dio {n_crudo}")
            detalle_listas.append({"pedidos": esperados, "dados": n_crudo})
            if n_crudo > n_rec and not ejemplo_lista_crudo:
                ejemplo_lista_crudo = crudo   # un caso donde el recorte perdio elementos
        muestrario.append((pregunta, r))

    # --- relevancia ---
    ok_rel = ok_rel_frase = total_rel = 0
    for pregunta, claves in RELEVANCIA:
        for k in range(args.muestras):
            r, _ = pedir(pregunta, k); todas.append(r)
            total_rel += 1
            claves_planas = [sin_tildes(c) for c in claves]
            ok = any(c in sin_tildes(r) for c in claves_planas)
            ok_rel += ok
            ok_rel_frase += any(c in sin_tildes(primera_frase(r)) for c in claves_planas)
            anotar("no habla del tema", pregunta, r, ok,
                   "deberia mencionar: " + ", ".join(claves))
        muestrario.append((pregunta, r))

    # --- conversaciones enteras: mide si aguanta el contexto acumulado ---
    from bpe_tokenizer import FIN_TURNO
    id_fin = id_fin_de_turno(tok)
    cierre = f"{FIN_TURNO}\n" if id_fin is not None else "\n\n"
    por_turno = {}          # numero de turno -> [juicios de idioma]
    ok_ident_conv = total_ident_conv = 0
    for k in range(args.muestras):
        for conversacion in CONVERSACIONES:
            contexto = ""
            for n, (pregunta, claves) in enumerate(conversacion, start=1):
                contexto += f"Usuario: {pregunta}\nAtlas:"
                torch.manual_seed(2000 + k * 13 + n)
                ids = tok.encode(contexto)[-bs:]
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                sal = model.generate(
                    idx, max_new_tokens=args.length, temperature=args.temperature,
                    top_k=args.top_k if args.top_k > 0 else None,
                    top_p=args.top_p if args.top_p < 1.0 else None,
                    repetition_penalty=args.repeticion, stop_id=id_fin,
                )[0].tolist()
                r = recortar_respuesta(tok.decode(sal[len(ids):]))
                todas.append(r)
                por_turno.setdefault(n, []).append(es_espanol(r))
                if claves:
                    total_ident_conv += 1
                    ok = any(c in sin_tildes(r) for c in claves)
                    ok_ident_conv += ok
                    anotar("identidad en conversacion", f"(turno {n}) {pregunta}", r, ok,
                           ", ".join(claves))
                juicio = es_espanol(r)
                if juicio is False:
                    anotar("responde en otro idioma", f"(turno {n}) {pregunta}", r, False, "español")
                contexto += f" {r}{cierre}"
            muestrario.append((" / ".join(p for p, _ in conversacion), r))

    espanol_por_turno = {}
    for n, juicios in sorted(por_turno.items()):
        d = [j for j in juicios if j is not None]
        espanol_por_turno[n] = (sum(d) / len(d) * 100 if d else float("nan"), len(d))

    # --- conversacion suelta (solo alimenta las metricas globales) ---
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
        "relevancia_primera_frase": ok_rel_frase / total_rel * 100,
        "identidad_en_conversacion": ok_ident_conv / total_ident_conv * 100 if total_ident_conv else float("nan"),
        "repeticion": repes,
        "parada_limpia": limpias,
        "respuestas_vacias": vacias,
        "palabras_por_respuesta": largo,
        "espanol_por_turno": {n: v[0] for n, v in espanol_por_turno.items()},
        "n_respuestas": len(todas),
    }

    print("=" * 62)
    print(f"{'EXAMEN DE ATLAS LUMERAK':^62}")
    print("=" * 62)
    # Cada medida con su intervalo: sin el, no hay forma de saber si una
    # diferencia entre dos modelos es real o es el azar del muestreo.
    filas = [
        ("Dice su nombre", aciertos_nombre, total_nombre, "+"),
        ("Dice quien lo creo", aciertos_creador, total_creador, "+"),
        ("Responde en espanol", sum(decidibles), len(decidibles), "+"),
        ("Obedece el numero pedido", ok_listas_crudo, total_listas, "+"),
        ("Habla del tema preguntado", ok_rel, total_rel, "+"),
        ("  ...en su primera frase", ok_rel_frase, total_rel, "+"),
        ("Identidad EN CONVERSACION", ok_ident_conv, total_ident_conv, "+"),
        ("Termina la frase", sum(termina_limpio(r) for r in todas), len(todas), "+"),
        ("Se repite", None, None, "-"),
        ("Respuestas vacias", sum(not r.strip() for r in todas), len(todas), "-"),
    ]
    for nombre, exitos, n, signo in filas:
        if exitos is None:
            print(f"  {nombre:26s} {repes:5.1f}%  {'':<16s} MENOS es mejor")
            continue
        pct = exitos / n * 100 if n else float("nan")
        a, b = intervalo(exitos, n)
        sentido = "mas es mejor" if signo == "+" else "MENOS es mejor"
        ancho = b - a
        aviso = "  <-- pocas muestras" if ancho > 20 else ""
        print(f"  {nombre:26s} {pct:5.1f}%  [{a:4.0f}-{b:3.0f}]  n={n:<4d} {sentido}{aviso}")

    print("\n  Los corchetes son el intervalo del 95%: donde esta el valor verdadero.")
    print("  Dos modelos solo son distintos si sus intervalos NO se solapan.")
    print("\n  En conversacion seguida, ¿sigue respondiendo en espanol?")
    for n, (pct, cuantas) in espanol_por_turno.items():
        print(f"    turno {n}: {pct:5.1f}%   (sobre {cuantas} respuestas juzgables)")
    if len(espanol_por_turno) > 1:
        primero = espanol_por_turno[1][0]
        ultimo = espanol_por_turno[max(espanol_por_turno)][0]
        if primero == primero and ultimo == ultimo and ultimo < primero - 10:
            print(f"    -> se degrada {primero - ultimo:.0f} puntos al acumularse el contexto")

    # Diagnostico de las listas: cuantos elementos pidio y cuantos dio.
    from collections import Counter
    print(f"\n  Listas: {res['obediencia_listas']:.0f}% acierta (lo que genera Atlas); "
          f"{res['obediencia_listas_tras_recorte']:.0f}% sobrevive al recorte del chat")
    reparto = Counter((d["pedidos"], d["dados"]) for d in detalle_listas)
    for (ped, dad), n in sorted(reparto.items()):
        marca = " <-- correcto" if ped == dad else ""
        print(f"    pidio {ped}, dio {dad:2d}  ({n} veces){marca}")

    cortas = sum(len(r.split()) <= 25 for r in todas) / len(todas) * 100
    print(f"\n  Longitud media: {largo:.0f} palabras por respuesta "
          f"({cortas:.0f}% son de 25 palabras o menos)")
    print(f"  {len(todas)} respuestas en {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.historial) or ".", exist_ok=True)
    with open(args.historial, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"\n  Guardado en {args.historial} (una linea por examen, para ver la curva)")

    # ---------------------------------------------------------------- errores
    # Lo que de verdad sirve para mejorar: no el porcentaje, sino QUE
    # contesto mal y a que. Se imprimen agrupados por tipo y se guardan
    # todos, para poder leerlos, entender el patron y corregirlo.
    if fallos:
        from collections import Counter, defaultdict
        conteo = Counter(f["categoria"] for f in fallos)
        por_cat = defaultdict(list)
        for f in fallos:
            por_cat[f["categoria"]].append(f)

        print("\n" + "=" * 62)
        print(f"{'ERRORES A CORREGIR':^62}")
        print("=" * 62)
        for categoria, n in conteo.most_common():
            print(f"\n[{categoria}]  {n} fallos")
            vistos = set()
            mostrados = 0
            for f in por_cat[categoria]:
                clave = (f["pregunta"], f["respuesta"][:60])
                if clave in vistos:
                    continue
                vistos.add(clave)
                print(f"  P: {f['pregunta']}")
                print(f"  R: {(f['respuesta'] or '(vacia)')[:220]}")
                if f["esperado"]:
                    print(f"     esperado: {f['esperado']}")
                mostrados += 1
                if mostrados >= args.errores:
                    break
            if n > mostrados:
                print(f"  ... y {n - mostrados} mas (todos en el archivo de errores)")

        ruta_fallos = os.path.join(os.path.dirname(args.historial) or ".", "errores.jsonl")
        with open(ruta_fallos, "w", encoding="utf-8") as f:
            for e in fallos:
                f.write(json.dumps({**e, "checkpoint": args.checkpoint,
                                    "fecha": res["fecha"]}, ensure_ascii=False) + "\n")
        print(f"\n  Los {len(fallos)} errores completos estan en {ruta_fallos}")

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
