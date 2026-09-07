"""
PASO 10: Preparar el corpus de INSTRUCCIONES, bien hecho esta vez.

Que cambia respecto de lo que hicimos antes (prepare_oasst2.py):

1) MASCARA DE PERDIDA. Antes, el entrenamiento calculaba el error sobre
   todos los tokens del lote, incluidos los de "Usuario: <pregunta>". O
   sea que la mitad del aprendizaje se gastaba en ensenarle a INVENTAR
   preguntas en vez de a responderlas -- y se notaba: el modelo escribia
   transcripciones de chat enteras, con preguntas incluidas, en vez de
   contestar la que le habiamos hecho. Aca cada token queda marcado con
   un 1 (cuenta para el error) o un 0 (no cuenta), y solo cuentan los que
   dice Atlas.

2) EJEMPLOS ALINEADOS. Antes se tomaban ventanas al azar del texto, que
   casi siempre empezaban a mitad de una respuesta. Aca cada ventana
   empieza en un "Usuario:" completo.

3) CONVERSACIONES DE VARIOS TURNOS. De oasst2 antes solo sacabamos pares
   pregunta-respuesta sueltos; se tiraba el resto del arbol. Aca se
   reconstruyen los dialogos completos, que es lo unico que ensena a
   seguir el hilo de una conversacion.

4) VARIAS FUENTES, elegibles. Ver --fuentes.

Salida: dos archivos binarios del mismo largo (tokens y mascara) mas un
.json con los datos de la corrida. Los consume train_sft.py.
"""

import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd

import identidad
from bpe_tokenizer import BPETokenizer, FIN_TURNO
from text_cleaning import es_espanol, limpiar_texto

HF = "https://huggingface.co/datasets"
URLS = {
    "oasst2_train": f"{HF}/OpenAssistant/oasst2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "oasst2_val": f"{HF}/OpenAssistant/oasst2/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "dolly_es": f"{HF}/argilla/databricks-dolly-15k-curated-multilingual/resolve/refs%2Fconvert%2Fparquet/default/es/0000.parquet",
    "alpaca_es": f"{HF}/bertin-project/alpaca-spanish/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
}
OPENHERMES = [
    f"{HF}/Iker/OpenHermes-2.5-Spanish/resolve/refs%2Fconvert%2Fparquet/default/train/{i:04d}.parquet"
    for i in range(4)
]

# Conversacion = lista de turnos (lo que dice el usuario, lo que responde Atlas).
Conversacion = list[tuple[str, str]]


def descargar(url: str, destino: str) -> str:
    if os.path.exists(destino):
        print(f"  (ya descargado: {destino})")
        return destino
    print(f"  descargando {os.path.basename(destino)}...", flush=True)
    urllib.request.urlretrieve(url, destino)
    return destino


# ---------------------------------------------------------------------
# Fuentes
# ---------------------------------------------------------------------
def cargar_oasst2(tmp: str) -> list[Conversacion]:
    """Reconstruye los dialogos completos del arbol, no solo pares sueltos.

    oasst2 no guarda conversaciones: guarda mensajes con un puntero a su
    mensaje padre. Un mismo mensaje puede tener varias respuestas, asi que
    el conjunto es un bosque de arboles. Cada camino de la raiz a una hoja
    es un dialogo valido; eso es lo que extraemos.
    """
    df = pd.concat(
        [pd.read_parquet(descargar(URLS["oasst2_train"], f"{tmp}/oasst2_train.parquet")),
         pd.read_parquet(descargar(URLS["oasst2_val"], f"{tmp}/oasst2_val.parquet"))],
        ignore_index=True,
    )
    es = df[(df["lang"] == "es") & (~df["deleted"]) & (df["review_result"] == True)]  # noqa: E712

    por_id = {r["message_id"]: r for _, r in es.iterrows()}
    hijos: dict[object, list[object]] = {}
    raices = []
    for mid, r in por_id.items():
        padre = r["parent_id"]
        if padre is None or padre not in por_id:
            # Solo sirve como raiz si es el usuario quien empieza.
            if r["role"] == "prompter":
                raices.append(mid)
        else:
            hijos.setdefault(padre, []).append(mid)

    conversaciones: list[Conversacion] = []

    def recorrer(mid, turnos_previos: Conversacion, pendiente: str | None):
        """Baja por el arbol acumulando turnos. `pendiente` es la pregunta del
        usuario que todavia no tiene respuesta emparejada."""
        r = por_id[mid]
        texto = str(r["text"]).strip()
        if r["role"] == "prompter":
            turnos, nuevo_pendiente = turnos_previos, texto
        else:
            if pendiente is None:
                return  # respuesta sin pregunta: rama invalida
            turnos, nuevo_pendiente = turnos_previos + [(pendiente, texto)], None

        descendencia = hijos.get(mid, [])
        if not descendencia:
            # Hoja: se guarda el dialogo si termina en una respuesta de Atlas.
            if nuevo_pendiente is None and turnos:
                conversaciones.append(turnos)
            return
        for h in descendencia:
            recorrer(h, turnos, nuevo_pendiente)

    for raiz in raices:
        recorrer(raiz, [], None)

    # Ramas hermanas comparten prefijo; el mismo dialogo puede aparecer
    # repetido si dos hojas distintas cuelgan del mismo camino.
    vistos, unicas = set(), []
    for c in conversaciones:
        clave = tuple(c)
        if clave not in vistos:
            vistos.add(clave)
            unicas.append(c)
    return unicas


def cargar_dolly(tmp: str) -> list[Conversacion]:
    """Dolly: escrito a mano por empleados de Databricks, traducido al espanol.
    Algunas preguntas vienen con un texto de contexto que hay que incluir."""
    df = pd.read_parquet(descargar(URLS["dolly_es"], f"{tmp}/dolly_es.parquet"))
    salida = []
    for _, r in df.iterrows():
        instruccion = str(r["instruction"]).strip()
        contexto = str(r["context"] or "").strip()
        respuesta = str(r["response"]).strip()
        if not instruccion or not respuesta:
            continue
        pregunta = f"{instruccion}\n\n{contexto}" if contexto else instruccion
        salida.append([(pregunta, respuesta)])
    return salida


def cargar_alpaca(tmp: str) -> list[Conversacion]:
    df = pd.read_parquet(descargar(URLS["alpaca_es"], f"{tmp}/alpaca_es.parquet"))
    salida = []
    for _, r in df.iterrows():
        instruccion = str(r["instruction"]).strip()
        entrada = str(r["input"] or "").strip()
        salida_txt = str(r["output"]).strip()
        if not instruccion or not salida_txt:
            continue
        pregunta = f"{instruccion}\n\n{entrada}" if entrada else instruccion
        salida.append([(pregunta, salida_txt)])
    return salida


def cargar_openhermes(tmp: str, limite: int) -> list[Conversacion]:
    """1 millon de instrucciones en espanol. Se procesa archivo por archivo y
    se borra cada uno al terminar: son ~900 MB en total."""
    salida: list[Conversacion] = []
    for url in OPENHERMES:
        if len(salida) >= limite:
            break
        ruta = descargar(url, f"{tmp}/openhermes_{os.path.basename(url)}")
        df = pd.read_parquet(ruta)
        for conv in df["conversations"]:
            turnos, pendiente = [], None
            for m in conv:
                quien, texto = m["from"], str(m["value"]).strip()
                if quien == "system":
                    continue  # no usamos mensajes de sistema en este formato
                if quien == "human":
                    pendiente = texto
                elif quien == "gpt" and pendiente:
                    turnos.append((pendiente, texto))
                    pendiente = None
            if turnos:
                salida.append(turnos)
                if len(salida) >= limite:
                    break
        del df
        os.remove(ruta)
        print(f"  {len(salida):,} conversaciones acumuladas", flush=True)
    return salida


CARGADORES = {
    "oasst2": lambda tmp, args: cargar_oasst2(tmp),
    "dolly": lambda tmp, args: cargar_dolly(tmp),
    "identidad": lambda tmp, args: identidad.ejemplos() * args.repetir_identidad,
    "alpaca": lambda tmp, args: cargar_alpaca(tmp),
    "openhermes": lambda tmp, args: cargar_openhermes(tmp, args.max_openhermes),
}


# ---------------------------------------------------------------------
# Conversacion -> tokens + mascara
# ---------------------------------------------------------------------
def codificar(conv: Conversacion, tok: BPETokenizer, largo_max: int):
    """Devuelve (tokens, mascara) o None si no entra ni recortando turnos.

    La mascara vale 1 en los tokens que Atlas tiene que aprender a producir
    -- su respuesta y el simbolo de fin de turno -- y 0 en los de la
    pregunta.

    El simbolo de fin de turno entra en la parte supervisada a proposito:
    aprender a emitirlo ES la tarea. Antes se usaba una linea en blanco como
    final, pero medido sobre los datos reales solo el 27% de las lineas en
    blanco marcaban un final; el otro 73% separaba parrafos dentro de la
    respuesta. Ese simbolo, en cambio, no significa ninguna otra cosa.

    Se codifica la pregunta y la respuesta por separado para saber donde
    cae exactamente la frontera. Da el mismo resultado que codificar todo
    junto porque el BPE parte el texto en palabras antes de fusionar, y
    " Claro" (con su espacio) es siempre una palabra aparte.
    """
    id_fin = tok.especiales[FIN_TURNO]
    salto = tok.encode("\n")     # separa un turno del siguiente, ya sin supervisar
    turnos = list(conv)
    while turnos:
        ids: list[int] = []
        mascara: list[int] = []
        for usuario, atlas in turnos:
            pregunta = limpiar_texto(f"Usuario: {usuario}\nAtlas:")
            respuesta = limpiar_texto(f" {atlas}")
            ip, ir = tok.encode(pregunta), tok.encode(respuesta)
            ids += ip + ir + [id_fin] + salto
            mascara += [0] * len(ip) + [1] * len(ir) + [1] + [0] * len(salto)
        if len(ids) <= largo_max:
            # Se devuelven como arrays de numpy y no como listas de Python:
            # con un millon de conversaciones la diferencia es de unos 3 GB
            # de memoria a unos 600 MB.
            return np.array(ids, dtype=np.uint16), np.array(mascara, dtype=np.uint8)
        if len(turnos) == 1:
            return None  # un solo intercambio y aun asi no entra
        turnos = turnos[1:]  # descartar el turno mas viejo y reintentar
    return None


def empaquetar(ejemplos, ventana: int):
    """Junta ejemplos completos dentro de cada ventana, sin partir ninguno.

    Cada ventana tiene `ventana` = block_size + 1 tokens: el modelo recibe
    los primeros block_size como entrada y los ultimos block_size como
    objetivo, desplazados en uno. Cuando el siguiente ejemplo no entra, se
    rellena el hueco con ceros marcados como "no cuenta" y se abre una
    ventana nueva. Asi toda ventana arranca en un "Usuario:" completo.
    """
    # Cada ventana lleva al menos un ejemplo, asi que nunca puede haber mas
    # ventanas que ejemplos: se reserva ese maximo de una vez y al final se
    # recorta. Reservar (en vez de ir agregando listas) es lo que hace que
    # esto funcione con un millon de conversaciones sin agotar la memoria.
    v_ids = np.zeros((len(ejemplos), ventana), dtype=np.uint16)
    v_mask = np.zeros((len(ejemplos), ventana), dtype=np.uint8)

    fila, usado = 0, 0
    for ids, mascara in ejemplos:
        if usado and usado + len(ids) > ventana:
            fila += 1          # el resto de la fila queda en cero: relleno
            usado = 0          # con mascara cero, o sea que no cuenta
        v_ids[fila, usado:usado + len(ids)] = ids
        v_mask[fila, usado:usado + len(ids)] = mascara
        usado += len(ids)
    n = fila + 1 if usado else fila
    return v_ids[:n], v_mask[:n]


# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepara el corpus de instrucciones con mascara de perdida.")
    parser.add_argument("--bpe", required=True,
                        help="EL MISMO tokenizador del pre-entrenamiento (atlas_bpe.json). "
                             "Con otro, los pesos ya entrenados no significarian nada.")
    parser.add_argument("--fuentes", nargs="+", default=["oasst2", "dolly", "identidad"],
                        choices=sorted(CARGADORES), help="Que datasets incluir.")
    parser.add_argument("--block_size", type=int, default=512,
                        help="Debe coincidir con el del modelo.")
    parser.add_argument("--out_dir", default="atlas_lumerak/data")
    parser.add_argument("--nombre", default="atlas_sft")
    parser.add_argument("--tmp", default="atlas_lumerak/data/_tmp_sft")
    parser.add_argument("--repetir_identidad", type=int, default=20,
                        help="Cuantas veces repetir los ejemplos de identidad. Son pocos y muy "
                             "importantes; repetirlos es la forma de que no se pierdan entre "
                             "cientos de miles de ejemplos ajenos.")
    parser.add_argument("--max_openhermes", type=int, default=200_000)
    parser.add_argument("--sin_filtro_idioma", action="store_true",
                        help="No descartar las conversaciones cuya respuesta no este en espanol.")
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)

    tok = BPETokenizer.load(args.bpe)
    antes = tok.vocab_size
    id_fin = tok.agregar_especial(FIN_TURNO)
    print(f"Tokenizador: {args.bpe} ({antes:,} simbolos)")
    if tok.vocab_size != antes:
        print(f"  + simbolo de fin de turno {FIN_TURNO!r} = numero {id_fin} "
              f"-> {tok.vocab_size:,} simbolos")
        print(f"  (va al final: los {antes:,} anteriores conservan su numero y su "
              f"significado para el modelo ya entrenado)")
    print()

    conversaciones: list[Conversacion] = []
    conteo_fuentes = {}
    for fuente in args.fuentes:
        print(f"[{fuente}]")
        nuevas = CARGADORES[fuente](args.tmp, args)
        conteo_fuentes[fuente] = len(nuevas)
        conversaciones += nuevas
        print(f"  -> {len(nuevas):,} conversaciones\n")

    # El corpus traducido conserva alrededor de un 1% de respuestas sin
    # traducir. Parece poco, pero es el unico ingles que Atlas ve etiquetado
    # como "asi se responde", y basta: en conversacion real el ingles
    # aparecia en 3 de cada 8 turnos. Un asistente en espanol no tiene por
    # que aprender de esas.
    if not args.sin_filtro_idioma:
        antes = len(conversaciones)
        conversaciones = [c for c in conversaciones
                          if all(es_espanol(a) is not False for _, a in c)]
        fuera = antes - len(conversaciones)
        print(f"Filtro de idioma: descartadas {fuera:,} conversaciones "
              f"({fuera / antes * 100:.1f}%) cuya respuesta no esta en espanol\n")

    print(f"Total: {len(conversaciones):,} conversaciones. Codificando...")
    ejemplos, descartadas = [], 0
    for i, conv in enumerate(conversaciones):
        r = codificar(conv, tok, args.block_size)
        if r is None:
            descartadas += 1
        else:
            ejemplos.append(r)
        if (i + 1) % 50_000 == 0:
            print(f"  {i + 1:,}/{len(conversaciones):,}", flush=True)
    print(f"Codificadas: {len(ejemplos):,} | descartadas por no entrar en "
          f"{args.block_size} tokens: {descartadas:,}")

    rng = np.random.default_rng(args.semilla)
    rng.shuffle(ejemplos)

    ventana = args.block_size + 1
    v_ids, v_mask = empaquetar(ejemplos, ventana)
    print(f"Ventanas de {ventana} tokens: {len(v_ids):,}")

    arr_ids = v_ids.reshape(-1)
    arr_mask = v_mask.reshape(-1)
    supervisados = int(arr_mask.sum())
    print(f"Tokens totales: {len(arr_ids):,} | supervisados (cuentan para el error): "
          f"{supervisados:,} ({supervisados / len(arr_ids) * 100:.1f}%)")

    base = os.path.join(args.out_dir, args.nombre)
    # El tokenizador ampliado se guarda junto al corpus: sin el, estos
    # numeros no se pueden traducir de vuelta a texto.
    tok.save(base + "_bpe.json")
    arr_ids.tofile(base + "_tokens.bin")
    arr_mask.tofile(base + "_mask.bin")
    meta = {
        "block_size": args.block_size,
        "ventana": ventana,
        "n_ventanas": len(v_ids),
        "vocab_size": tok.vocab_size,
        "bpe": os.path.basename(base + "_bpe.json"),
        "fin_de_turno": id_fin,
        "fuentes": conteo_fuentes,
        "ejemplos": len(ejemplos),
        "descartados": descartadas,
        "tokens_totales": int(len(arr_ids)),
        "tokens_supervisados": supervisados,
    }
    with open(base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nGuardado:\n  {base}_tokens.bin\n  {base}_mask.bin\n  {base}_meta.json"
          f"\n  {base}_bpe.json  (tokenizador ampliado -- usa ESTE de aqui en adelante)")


if __name__ == "__main__":
    main()
