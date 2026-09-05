"""
La arquitectura del Transformer de Atlas Lumerak (Pasos 3 y 4), ahora
reutilizable: en vez de tamanos fijos "quemados" en el archivo, cada
pieza recibe sus dimensiones como parametros. Esto nos permite usar un
modelo chico para probar rapido en CPU, y uno mas grande para entrenar
en serio con GPU, sin duplicar codigo.

Version con atencion optimizada (Flash Attention / scaled_dot_product_attention):
en vez de calcular cada cabeza de atencion por separado en un bucle de
Python, se hace una sola proyeccion combinada para Q, K y V, y se usa la
funcion optimizada de PyTorch que aprovecha kernels fusionados en GPU
(Flash Attention en hardware compatible). Matematicamente calcula
exactamente lo mismo que antes -- verificado numericamente que produce
resultados identicos (diferencia de ~1e-7, puro ruido de redondeo) --
solo que mas rapido. Ver migrate_checkpoint.py para pasar un checkpoint
entrenado con la version anterior a esta.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_size).transpose(1, 2)

        # is_causal=True aplica la misma mascara triangular de antes (no
        # ver el futuro), pero fusionada dentro del kernel optimizado en
        # vez de una mascara manual con -infinito + softmax por separado.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class FeedForward(nn.Module):
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
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 128,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        if tie_weights:
            # La tabla de entrada (letra -> vector) y la de salida
            # (vector -> letra) resuelven el mismo problema en direcciones
            # opuestas. Compartir una sola matriz ahorra parametros y suele
            # mejorar la calidad en modelos chicos. Es lo que hace GPT-2.
            #
            # Por defecto viene desactivado para no romper checkpoints
            # anteriores: si un checkpoint sin pesos atados se cargara en un
            # modelo con pesos atados, las dos tablas se pisarian entre si
            # en silencio, sin ningun error visible.
            self.lm_head.weight = self.token_embedding_table.weight

        # Inicializacion al estilo GPT-2. Por defecto, PyTorch inicializa
        # los embeddings con desviacion 1.0, ~50 veces mas grande que el
        # resto de la red; eso desbalancea el flujo residual desde el
        # primer paso y hace que el entrenamiento arranque mas lento.
        self.apply(self._init_weights)

        # Las capas que escriben de vuelta al flujo residual se inicializan
        # mas chicas todavia, en proporcion a la profundidad: con 12 bloques
        # sumando al mismo flujo, sin esto la senal se acumula y crece
        # descontroladamente con la profundidad.
        for nombre, p in self.named_parameters():
            if nombre.endswith("proj.weight") or nombre.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(modulo):
        if isinstance(modulo, nn.Linear):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            if modulo.bias is not None:
                nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, nn.Embedding):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t, device=device))
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """
        temperature: que tan "arriesgado" es al elegir. Menor a 1 hace que
            prefiera lo mas probable (mas coherente, menos creativo); mayor
            a 1 lo hace mas impredecible. 1.0 = sin cambios.
        top_k: si se indica, solo considera los k caracteres mas probables
            y descarta el resto. Evita que de vez en cuando elija una letra
            absurda de la "cola larga" y descarrile la frase entera.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                umbral = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < umbral, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx
