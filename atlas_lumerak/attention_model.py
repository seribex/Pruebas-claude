"""
PASO 3: Atencion (el mecanismo detras de los Transformers, y de Claude).

El modelo del Paso 2 solo miraba 1 letra para adivinar la siguiente. Aca
el modelo puede mirar TODAS las letras anteriores (hasta BLOCK_SIZE) y
decidir, para cada una, cuanto le importa a la hora de predecir. Eso es
"atencion": cada posicion le pregunta a las posiciones anteriores "que tan
relevante eres para mi ahora mismo", y arma su respuesta como un promedio
pesado por esas relevancias.

Ademas usamos varias "cabezas" de atencion en paralelo (MultiHeadAttention)
-- la idea que se te ocurrio antes: en vez de 1 sola via haciendo todo el
trabajo, varias vias mas simples, cada una fijandose en un tipo de relacion
distinto, y al final se combinan.
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

BLOCK_SIZE = 8    # cuantas letras de contexto puede mirar el modelo
BATCH_SIZE = 4
N_EMBD = 32       # tamano del vector que representa a cada letra
N_HEADS = 4       # cuantas cabezas de atencion en paralelo


def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


class Head(nn.Module):
    """Una sola cabeza de atencion."""

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # tril = una mascara triangular: impide que una letra "vea" letras
        # futuras que todavia no deberia conocer al predecir.
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)    # que informacion "ofrece" cada posicion
        q = self.query(x)  # que informacion "busca" cada posicion
        v = self.value(x)  # la informacion real que se pasa si hay match

        # que tanto le interesa a cada posicion cada otra posicion anterior
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        return wei @ v


class MultiHeadAttention(nn.Module):
    """Varias cabezas de atencion en paralelo, combinadas al final."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class AttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.sa_heads = MultiHeadAttention(N_HEADS, N_EMBD // N_HEADS)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape

        tok_emb = self.token_embedding_table(idx)                       # (b, t, N_EMBD)
        pos_emb = self.position_embedding_table(torch.arange(t))        # (t, N_EMBD)
        x = tok_emb + pos_emb                                            # cada letra + su posicion
        x = self.sa_heads(x)                                             # aca ocurre la atencion
        logits = self.lm_head(x)                                         # (b, t, vocab_size)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]  # el modelo solo puede mirar BLOCK_SIZE letras atras
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    model = AttentionLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
