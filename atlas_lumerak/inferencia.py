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

MARCA_USUARIO = "\nUsuario:"


def recortar_respuesta(texto: str, parar_en_blanco: bool = True) -> str:
    """Se queda con la respuesta y descarta la inercia que viene despues."""
    # Si empieza a inventarse el turno del usuario, ahi termino seguro.
    texto = texto.split(MARCA_USUARIO)[0]

    if parar_en_blanco:
        # Primera linea en blanco = fin de la respuesta. Se pierde el caso
        # legitimo de una respuesta de varios parrafos, pero hoy Atlas no
        # produce ninguna que valga la pena mantener entera.
        partes = texto.strip().split("\n\n")
        texto = partes[0] if partes[0].strip() else texto

    return texto.strip()
