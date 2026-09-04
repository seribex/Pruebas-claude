"""
PASO 1: Tokenizador de caracteres.

Una red neuronal no entiende texto, solo numeros. El tokenizador es el
puente: convierte texto en una lista de numeros (encode) y esa lista de
numeros de vuelta en texto (decode).

Empezamos con la version mas simple posible: un "vocabulario" formado por
cada caracter distinto que aparece en el texto de entrenamiento (letras,
espacios, puntuacion...). A cada caracter le asignamos un numero unico.

Ejemplo: si el texto solo tuviera "abc", el vocabulario seria {a, b, c} y
"cab" se convertiria en [2, 0, 1].
"""


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)


if __name__ == "__main__":
    with open("atlas_lumerak/data/sample.txt", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    print(f"Tamano del vocabulario: {tokenizer.vocab_size} caracteres distintos")
    print(f"Vocabulario: {sorted(tokenizer.char_to_id.keys())}")

    ejemplo = "Atlas Lumerak"
    ids = tokenizer.encode(ejemplo)
    de_vuelta = tokenizer.decode(ids)

    print(f"\nTexto original: {ejemplo!r}")
    print(f"Codificado (numeros): {ids}")
    print(f"Decodificado de vuelta: {de_vuelta!r}")
