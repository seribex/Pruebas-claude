"""
PASO 2: El modelo mas simple posible - un "modelo de bigramas".

Este modelo solo mira UN caracter para adivinar el siguiente. No tiene
capa oculta, ni atencion, ni nada elaborado: es literalmente una tabla
gigante que dice "despues de la letra X, que tan probable es cada letra
del vocabulario". Es deliberadamente ingenuo: el objetivo de este paso no
es que escriba bien, sino ver el ciclo COMPLETO funcionando de punta a
punta (datos -> modelo -> entrenamiento -> generacion). Los pasos
siguientes (atencion, transformer) son versiones mas listas de este mismo
ciclo.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import CharTokenizer

torch.manual_seed(42)

with open("atlas_lumerak/data/sample.txt", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

BLOCK_SIZE = 8   # cuantos caracteres de contexto usamos por muestra
BATCH_SIZE = 4   # cuantas muestras procesamos a la vez


def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Una tabla vocab_size x vocab_size: la fila i son los "puntajes"
        # (logits) de que tan probable es cada letra como siguiente,
        # dado que la letra actual es la i-esima del vocabulario.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (batch, tiempo, vocab_size)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # solo importa la ultima posicion
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    model = BigramLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    start = torch.zeros((1, 1), dtype=torch.long)

    print("--- Antes de entrenar (pesos al azar) ---")
    print(repr(tokenizer.decode(model.generate(start, max_new_tokens=80)[0].tolist())))

    steps = 3000
    for step in range(steps):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == steps - 1:
            print(f"paso {step:4d} | perdida de entrenamiento: {loss.item():.4f}")

    print("\n--- Despues de entrenar ---")
    print(repr(tokenizer.decode(model.generate(start, max_new_tokens=200)[0].tolist())))
