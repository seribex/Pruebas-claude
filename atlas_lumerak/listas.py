"""
Ensenar a Atlas a dar EXACTAMENTE el numero de cosas que se le piden.

Por que hace falta generarlo: medido sobre el corpus real, de 18,395
conversaciones solo 238 piden un numero explicito de cosas, y solo 44 traen
una lista contable en la respuesta. Es el 0.24%. Extrapolado al millon de
conversaciones son unos 2,400 ejemplos enterrados entre un millon: no
alcanza para aprender una habilidad como contar.

Y se nota: pidiendole 3 frutas daba 5, pidiendole 5 daba 3. El numero de la
pregunta sencillamente no controlaba la salida, porque nada le enseno que
tuviera que hacerlo.

SOBRE MEDIR ESTO CON HONESTIDAD: las categorias marcadas como reservadas NO
se usan para entrenar. Estan en el examen a proposito, para que la nota mida
si Atlas aprendio A CONTAR y no si memorizo estas listas. Si acierta con las
entrenadas y falla con las reservadas, no aprendio nada: nos hizo trampa, y
nos la habriamos hecho nosotros mismos al elegir los datos.
"""

import unicodedata

CATEGORIAS: dict[str, list[str]] = {
    "frutas": ["manzana", "plátano", "naranja", "pera", "uva", "fresa", "sandía",
               "melón", "piña", "mango", "kiwi", "cereza", "durazno", "limón"],
    "colores": ["rojo", "azul", "verde", "amarillo", "negro", "blanco", "morado",
                "naranja", "rosa", "gris", "marrón", "celeste"],
    "animales": ["perro", "gato", "caballo", "vaca", "león", "tigre", "elefante",
                 "conejo", "oso", "lobo", "zorro", "ballena", "águila", "delfín"],
    "países": ["Perú", "México", "España", "Argentina", "Chile", "Colombia",
               "Brasil", "Francia", "Italia", "Japón", "Canadá", "Portugal"],
    "verduras": ["zanahoria", "lechuga", "tomate", "cebolla", "brócoli", "espinaca",
                 "papa", "pepino", "calabaza", "apio", "pimiento", "col"],
    "bebidas": ["agua", "café", "té", "jugo de naranja", "leche", "limonada",
                "chocolate caliente", "batido", "refresco", "infusión"],
    "flores": ["rosa", "girasol", "margarita", "tulipán", "clavel", "orquídea",
               "lirio", "jazmín", "amapola", "violeta"],
    "árboles": ["pino", "roble", "sauce", "palmera", "cedro", "olivo", "abeto",
                "álamo", "eucalipto", "naranjo"],
    "muebles": ["mesa", "silla", "sofá", "cama", "estantería", "armario",
                "escritorio", "banco", "cómoda", "taburete"],
    "prendas de ropa": ["camisa", "pantalón", "chaqueta", "falda", "vestido",
                        "abrigo", "bufanda", "gorro", "calcetines", "suéter"],
    "medios de transporte": ["auto", "bicicleta", "autobús", "tren", "avión",
                             "barco", "motocicleta", "metro", "tranvía", "camión"],
    "herramientas": ["martillo", "destornillador", "sierra", "llave inglesa",
                     "taladro", "alicate", "cincel", "lima", "nivel", "cinta métrica"],
    "instrumentos de cocina": ["cuchillo", "sartén", "olla", "cuchara", "tenedor",
                               "batidora", "colador", "tabla de cortar", "rallador", "espátula"],
    "materiales de estudio": ["cuaderno", "lápiz", "bolígrafo", "regla", "goma de borrar",
                              "mochila", "calculadora", "resaltador", "carpeta", "diccionario"],
    "estaciones del año": ["primavera", "verano", "otoño", "invierno"],
    "días de la semana": ["lunes", "martes", "miércoles", "jueves", "viernes",
                          "sábado", "domingo"],
    "meses del año": ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                      "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
    "planetas del sistema solar": ["Mercurio", "Venus", "la Tierra", "Marte",
                                   "Júpiter", "Saturno", "Urano", "Neptuno"],
    "números pares": ["2", "4", "6", "8", "10", "12", "14", "16", "18", "20"],
    "figuras geométricas": ["círculo", "cuadrado", "triángulo", "rectángulo",
                            "rombo", "pentágono", "hexágono", "óvalo"],
    "partes del cuerpo": ["cabeza", "brazo", "pierna", "mano", "pie", "ojo",
                          "oreja", "nariz", "boca", "espalda"],
    "idiomas": ["español", "inglés", "francés", "portugués", "italiano", "alemán",
                "japonés", "chino", "ruso", "árabe"],
    "postres": ["helado", "pastel", "flan", "gelatina", "galletas", "brownie",
                "arroz con leche", "mousse", "tarta de manzana", "churros"],
    "aves": ["gorrión", "paloma", "águila", "loro", "búho", "colibrí", "cóndor",
             "flamenco", "pingüino", "cisne"],
}

# Para n=1 hay que decir "dame una fruta", no "dame 1 frutas". Entrenar con
# espanol mal escrito seria peor que no tener el ejemplo.
SINGULAR = {
    "frutas": "fruta", "colores": "color", "animales": "animal", "países": "país",
    "verduras": "verdura", "bebidas": "bebida", "flores": "flor", "árboles": "árbol",
    "muebles": "mueble", "prendas de ropa": "prenda de ropa",
    "medios de transporte": "medio de transporte", "herramientas": "herramienta",
    "instrumentos de cocina": "instrumento de cocina",
    "materiales de estudio": "material de estudio", "estaciones del año": "estación del año",
    "días de la semana": "día de la semana", "meses del año": "mes del año",
    "planetas del sistema solar": "planeta del sistema solar", "números pares": "número par",
    "figuras geométricas": "figura geométrica", "partes del cuerpo": "parte del cuerpo",
    "idiomas": "idioma", "postres": "postre", "aves": "ave",
}

# NO se entrenan. Estan en el examen para medir si aprendio a contar de
# verdad o solo memorizo las listas que le dimos.
RESERVADAS = {"instrumentos musicales", "profesiones", "ríos", "deportes"}

PLANTILLAS = [
    "Escribe una lista de {n} {c}", "Dame {n} {c}", "Nombra {n} {c}",
    "Menciona {n} {c}", "Dime {n} {c}", "Enumera {n} {c}",
    "Escribe {n} {c}", "Necesito {n} {c}", "¿Puedes darme {n} {c}?",
    "Hazme una lista de {n} {c}", "Dame una lista con {n} {c}",
    "Quiero {n} {c}", "Escríbeme {n} {c}", "Anota {n} {c}",
]
INTROS = ["", "", "", "Claro, aquí tienes {n} {c}:\n", "Aquí van {n} {c}:\n",
          "Por supuesto:\n", "Claro:\n"]
PALABRA = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco",
           6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez"}


def _informal(texto: str) -> str:
    sin = "".join(c for c in unicodedata.normalize("NFD", texto)
                  if unicodedata.category(c) != "Mn" or c == "̃")
    sin = sin.replace("̃", "")
    sin = unicodedata.normalize("NFC", sin).lstrip("¿¡").rstrip("?!")
    return sin[:1].lower() + sin[1:] if sin else sin


def ejemplos(repeticiones: int = 1) -> list[list[tuple[str, str]]]:
    """Genera preguntas del tipo "dame N cosas" con respuestas de EXACTAMENTE N."""
    salida = []
    categorias = [(c, items) for c, items in CATEGORIAS.items() if c not in RESERVADAS]
    for rep in range(repeticiones):
        for ci, (categoria, items) in enumerate(categorias):
            for n in range(1, min(10, len(items)) + 1):
                plantilla = PLANTILLAS[(ci + n + rep) % len(PLANTILLAS)]
                # El numero, unas veces en cifra y otras en palabra.
                escrito = str(n) if (ci + n + rep) % 3 else PALABRA[n]
                # Concordancia: "una fruta" en singular, "3 frutas" en plural.
                nombre = SINGULAR[categoria] if n == 1 else categoria
                if n == 1:
                    escrito = "una" if SINGULAR[categoria][-1] == "a" else "un"
                pregunta = plantilla.format(n=escrito, c=nombre)
                if (ci + rep) % 4 == 0:
                    pregunta = _informal(pregunta)

                # Los elementos se toman rotando, para que no siempre salgan
                # los mismos primeros de la lista.
                inicio = (ci * 3 + n + rep) % len(items)
                elegidos = [items[(inicio + k) % len(items)] for k in range(n)]

                intro = INTROS[(ci + n + rep) % len(INTROS)].format(n=escrito, c=nombre)
                if (ci + n) % 5 == 0:
                    cuerpo = "\n".join(f"- {e}" for e in elegidos)
                else:
                    cuerpo = "\n".join(f"{i}. {e}" for i, e in enumerate(elegidos, 1))
                salida.append([(pregunta, intro + cuerpo)])
    return salida


if __name__ == "__main__":
    for reps in (1, 5, 20):
        e = ejemplos(reps)
        print(f"repeticiones={reps:3d} -> {len(e):6,} ejemplos | "
              f"{len({tuple(c) for c in e}):6,} distintos")
    print(f"\n{len(CATEGORIAS)} categorias, {len(CATEGORIAS) - len(RESERVADAS & set(CATEGORIAS))} "
          f"se entrenan")
    print(f"reservadas para el examen (NO se entrenan): {', '.join(sorted(RESERVADAS))}\n")
    for c in ejemplos(1)[:3] + ejemplos(1)[40:42]:
        p, r = c[0]
        print(f"  P: {p}\n  R: {r}\n")
