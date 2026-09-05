"""
PASO 7: Tokenizador BPE (Byte Pair Encoding).

Hasta ahora Atlas leia letra por letra. "ciudad" eran 6 decisiones
separadas, y una parte enorme de sus conexiones se gastaba simplemente
en aprender a deletrear.

BPE arma un vocabulario de *fragmentos* de palabras. La idea es simple y
se puede explicar en una linea: empieza con letras sueltas, y repite
muchas veces "busca el par de simbolos vecinos mas frecuente en todo el
texto, y fusionalos en un simbolo nuevo".

Ejemplo de como evoluciona:
    c,i,u,d,a,d   ->  ci,u,d,a,d  ->  ciu,da,d  ->  ciudad
Los fragmentos frecuentes ("ciudad", "cion", " de") terminan siendo un
solo simbolo; los raros siguen deletreandose. Es lo mismo que usan los
modelos grandes de verdad.

Beneficios concretos para Atlas:
  - Cada paso de entrenamiento cubre ~4-5x mas texto
  - Su memoria de contexto rinde ~4-5x mas
  - No malgasta capacidad aprendiendo ortografia; la usa en significado
"""

import json
import re
from collections import Counter, defaultdict

# Separa el texto en "palabras" antes de fusionar, para que BPE no cree
# fragmentos que crucen de una palabra a otra. El espacio se pega a la
# palabra que le sigue (" casa" es un simbolo distinto de "casa"), que es
# lo que permite reconstruir el texto exactamente al decodificar.
PATRON = re.compile(r" ?\w+| ?[^\s\w]+|\s+")


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[str, str], str] = {}
        # Orden en que se aprendio cada fusion. Se guarda aparte para
        # poder consultarlo en tiempo constante al codificar (buscarlo
        # recorriendo la lista de fusiones haria la codificacion
        # lentisima con vocabularios grandes).
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ---------------------------------------------------------------
    def train(self, texto: str, vocab_size: int, min_freq: int = 2, verbose: bool = True) -> None:
        # 1) Contar cuantas veces aparece cada palabra. Trabajar sobre
        #    palabras unicas (y su frecuencia) en vez de sobre todo el
        #    texto hace esto miles de veces mas rapido.
        frecuencias = Counter(PATRON.findall(texto))
        if verbose:
            print(f"Palabras distintas: {len(frecuencias):,}")

        # Las palabras que aparecen una sola vez (nombres propios raros,
        # numeros sueltos) son casi la mitad del total y no influyen en
        # que fusiones conviene aprender, pero si hacen el entrenamiento
        # mucho mas lento. Se excluyen de esta etapa; despues igual se
        # pueden codificar sin problema, en pedazos mas chicos.
        todos_los_caracteres = sorted({c for p in frecuencias for c in p})
        frecuentes = {p: n for p, n in frecuencias.items() if n >= min_freq}
        if verbose:
            print(f"Palabras usadas para aprender fusiones (>= {min_freq} apariciones): {len(frecuentes):,}")

        palabras = [tuple(p) for p in frecuentes]
        pesos = list(frecuentes.values())

        # 2) Vocabulario base: todos los caracteres del texto, incluidos
        #    los que solo aparecen en palabras raras (si no, esas palabras
        #    no se podrian codificar despues).
        caracteres = todos_los_caracteres
        vocabulario = list(caracteres)
        if verbose:
            print(f"Caracteres base: {len(caracteres)}")

        # 3) Indice de pares -> en que palabras aparecen. Sirve para no
        #    recorrer todas las palabras en cada fusion, solo las
        #    afectadas.
        conteo_pares: Counter = Counter()
        pares_en: defaultdict[tuple[str, str], set[int]] = defaultdict(set)

        def indexar(i: int, delta: int) -> None:
            palabra, peso = palabras[i], pesos[i]
            for par in zip(palabra, palabra[1:]):
                conteo_pares[par] += delta * peso
                if delta > 0:
                    pares_en[par].add(i)

        for i in range(len(palabras)):
            indexar(i, +1)

        # 4) Fusionar el par mas frecuente, una y otra vez.
        objetivo = vocab_size - len(caracteres)
        for paso in range(objetivo):
            if not conteo_pares:
                break
            par, cuenta = max(conteo_pares.items(), key=lambda kv: kv[1])
            if cuenta <= 0:
                break

            nuevo = par[0] + par[1]
            self.merges[par] = nuevo
            self.merge_ranks[par] = len(self.merge_ranks)
            vocabulario.append(nuevo)

            for i in list(pares_en[par]):
                palabra = palabras[i]
                if len(palabra) < 2:
                    continue
                indexar(i, -1)  # quitar los pares viejos de esta palabra

                # Aplicar la fusion dentro de la palabra
                nueva, j = [], 0
                while j < len(palabra):
                    if j < len(palabra) - 1 and (palabra[j], palabra[j + 1]) == par:
                        nueva.append(nuevo)
                        j += 2
                    else:
                        nueva.append(palabra[j])
                        j += 1
                palabras[i] = tuple(nueva)
                indexar(i, +1)  # registrar los pares nuevos

            del conteo_pares[par]
            del pares_en[par]

            if verbose and (paso + 1) % 1000 == 0:
                print(f"  fusiones: {paso + 1:,}/{objetivo:,} | ultimo: {nuevo!r}")

        self.token_to_id = {t: i for i, t in enumerate(vocabulario)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self._cache.clear()
        if verbose:
            print(f"Vocabulario final: {self.vocab_size:,} simbolos")

    # ---------------------------------------------------------------
    def _tokenizar_palabra(self, palabra: str) -> list[int]:
        if palabra in self._cache:
            return self._cache[palabra]

        piezas = [c for c in palabra if c in self.token_to_id]
        while len(piezas) >= 2:
            # De todas las fusiones posibles, aplicar la que se aprendio
            # primero (las primeras son las mas frecuentes).
            candidatos = [
                (self.merge_ranks[par], idx)
                for idx, par in enumerate(zip(piezas, piezas[1:]))
                if par in self.merge_ranks
            ]
            if not candidatos:
                break
            _, idx = min(candidatos)
            piezas[idx:idx + 2] = [self.merges[(piezas[idx], piezas[idx + 1])]]

        ids = [self.token_to_id[p] for p in piezas if p in self.token_to_id]
        self._cache[palabra] = ids
        return ids

    def encode(self, texto: str) -> list[int]:
        ids: list[int] = []
        for palabra in PATRON.findall(texto):
            ids.extend(self._tokenizar_palabra(palabra))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids if i in self.id_to_token)

    # ---------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocabulario": [t for t, _ in sorted(self.token_to_id.items(), key=lambda kv: kv[1])],
                    "merges": [[a, b] for (a, b) in self.merges],
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            datos = json.load(f)
        tok = cls()
        tok.token_to_id = {t: i for i, t in enumerate(datos["vocabulario"])}
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        tok.merges = {(a, b): a + b for a, b in datos["merges"]}
        tok.merge_ranks = {(a, b): i for i, (a, b) in enumerate(datos["merges"])}
        return tok
