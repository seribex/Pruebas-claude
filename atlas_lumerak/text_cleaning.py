"""
Limpieza de texto compartida por todos los scripts que preparan datos.

Vive en un solo lugar a proposito: antes estaba duplicada, y cuando
encontramos un bug (algunos simbolos y emojis tienen "LATIN" en su
nombre Unicode sin ser letras) hubo que arreglarlo en dos sitios.
"""

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
