"""
Limpieza de texto compartida por todos los scripts que preparan datos.

Vive en un solo lugar a proposito: antes estaba duplicada, y cuando
encontramos un bug (algunos simbolos y emojis tienen "LATIN" en su
nombre Unicode sin ser letras) hubo que arreglarlo en dos sitios.
"""

import re
import unicodedata

PUNTUACION_EXTRA = "¡¿«»—–…‘’“”"


def keep_char(c: str) -> bool:
    """Deja pasar texto latino normal y puntuacion en espanol; descarta
    escrituras extranjeras (chino, cirilico, etc.), simbolos sueltos y
    caracteres invisibles."""
    if c.isascii():
        return True
    # Ojo: algunos simbolos ("LATIN CROSS", letras dentro de circulos)
    # tienen "LATIN" en su nombre Unicode sin ser letras -- por eso se
    # exige ademas que la categoria Unicode sea de tipo Letra (L*).
    if unicodedata.category(c).startswith("L") and "LATIN" in unicodedata.name(c, ""):
        return True
    return c in PUNTUACION_EXTRA


def limpiar_texto(texto: str) -> str:
    return "".join(c for c in texto if keep_char(c))


# Palabras muy frecuentes de cada idioma. Sirven para saber en cual esta
# escrito un texto sin depender de ninguna libreria externa.
PAL_ES = set("de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este sí porque esta cuando muy sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros".split())
PAL_EN = set("the of and to in is that it for with was are you your this be have not on as at by from or an we they he she his her their there but what all can will one would about which when who been has had do does did if my me so no out up".split())


def es_espanol(texto: str) -> bool | None:
    """True, False, o None si el texto es demasiado corto para decidir."""
    pal = re.findall(r"[a-záéíóúñü]+", texto.lower())
    if len(pal) < 6:
        return None
    ne = sum(p in PAL_EN for p in pal)
    nes = sum(p in PAL_ES for p in pal)
    if ne == 0 and nes == 0:
        return None
    return nes >= ne
