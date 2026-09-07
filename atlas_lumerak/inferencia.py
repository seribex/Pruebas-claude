"""
Utilidades compartidas para hablar con un Atlas ya entrenado.

El problema que resuelve esto: Atlas no tiene una senal explicita de "aqui
termina mi respuesta". Se le enseno que sus respuestas acaban en una linea
en blanco, pero medido sobre los datos de entrenamiento, de cada 1093
lineas en blanco solo 300 marcaban el final -- las otras 793 estaban DENTRO
de una respuesta, separando parrafos. O sea que le dimos una senal que el
73% de las veces significa lo contrario de lo que queriamos.

Consecuencia visible: responde bien y despues sigue escribiendo por inercia,
divagando o repitiendose, hasta agotar los tokens que le diste.

Mientras no exista un token dedicado de fin de turno (el arreglo de fondo),
cortar en la primera linea en blanco recupera lo bueno y descarta el resto.
"""

import re

from bpe_tokenizer import FIN_TURNO

MARCA_USUARIO = "\nUsuario:"
# Un bloque que empieza en "1." , "2)" , "-" o "*" es la continuacion de una
# lista, no un parrafo nuevo. Se exige el espacio y algo detras para no
# confundirlo con un ano ("2020 fue...") o una hora.
ELEMENTO_DE_LISTA = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s+\S")


def id_fin_de_turno(tok) -> int | None:
    """Numero del simbolo de fin de turno, o None si el modelo es anterior a
    que existiera. Devolver None hace que todo siga funcionando igual que
    antes con los checkpoints viejos."""
    return getattr(tok, "especiales", {}).get(FIN_TURNO)


def recortar_respuesta(texto: str, parar_en_blanco: bool = True) -> str:
    """Se queda con la respuesta y descarta la inercia que viene despues."""
    # Si empieza a inventarse el turno del usuario, ahi termino seguro.
    # El simbolo de fin de turno es la senal buena: si esta, no hay nada
    # que adivinar. Todo lo de abajo son muletas para los modelos que no lo
    # tienen (el Atlas entrenado antes de este cambio).
    if FIN_TURNO in texto:
        return texto.split(FIN_TURNO)[0].strip()

    texto = texto.split(MARCA_USUARIO)[0]
    if not parar_en_blanco:
        return texto.strip()

    bloques = texto.strip().split("\n\n")
    if not bloques or not bloques[0].strip():
        return texto.strip()

    # La primera linea en blanco marca el final de la respuesta... salvo
    # cuando lo que sigue es otro elemento de la misma lista. Escribir
    # "1. uno\n\n2. dos\n\n3. tres" es markdown normal y era como venia
    # buena parte del corpus de entrenamiento; cortando en la primera linea
    # en blanco, una lista de tres elementos se quedaba en uno.
    # Medido sobre el modelo real: el acierto en "dame N cosas" pasaba del
    # 24% al 4% por culpa de este recorte, no por culpa del modelo.
    salida = [bloques[0]]
    for b in bloques[1:]:
        if not b.strip():
            continue          # varias lineas en blanco seguidas: no rompe la lista
        if not ELEMENTO_DE_LISTA.match(b):
            break
        salida.append(b)

    return "\n\n".join(salida).strip()
