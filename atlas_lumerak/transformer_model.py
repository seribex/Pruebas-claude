"""
PASO 4: El Transformer completo.

Al modelo del Paso 3 (atencion multi-cabeza) le agregamos dos piezas que
le faltan para ser un Transformer de verdad:

1. FeedForward: despues de que la atencion "recolecta" informacion de las
   letras anteriores, esta pieza le da al modelo espacio para procesar esa
   informacion antes de pasarla a la siguiente capa. La atencion mezcla
   informacion ENTRE posiciones; el feed-forward piensa sobre cada
   posicion POR SEPARADO.

2. Conexiones residuales (x = x + algo(x)) y normalizacion (LayerNorm):
   sin esto, apilar varias capas de atencion hace que el entrenamiento se
   vuelva inestable o directamente deje de aprender. La conexion residual
   es como una "autopista" por la que la informacion original siempre
   puede pasar de largo, incluso si una capa todavia no aprendio nada util;
   la normalizacion mantiene los numeros en un rango estable capa tras
   capa.

Un "Block" agrupa (atencion + feed-forward) con sus conexiones residuales.
Apilar varios Blocks es, literalmente, lo que hace "profunda" a una red
Transformer como la que usa Claude (a una escala miles de veces mayor).
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

BLOCK_SIZE = 8
BATCH_SIZE = 4
N_EMBD = 32
N_HEADS = 4
N_LAYER = 3   # cuantos "Blocks" apilamos


def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Procesa cada posicion por separado, tras la atencion."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Un bloque de Transformer: atencion + feed-forward, con residuales."""

    def __init__(self, n_embd: int, n_heads: int):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # "x +" es la conexion residual: la informacion original se suma
        # de vuelta, en vez de reemplazarse por completo en cada capa.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEADS) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    model = TransformerLanguageModel(tokenizer.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros totales del modelo: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start = torch.zeros((1, 1), dtype=torch.long)

    print("\n--- Antes de entrenar (pesos al azar) ---")
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
