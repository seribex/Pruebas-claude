"""
Migra un checkpoint entrenado con la arquitectura VIEJA (cabezas de
atencion separadas) a la arquitectura NUEVA (scaled_dot_product_attention
/ Flash Attention), reacomodando los pesos sin perder nada de lo
aprendido -- verificado matematicamente (ver conversacion del proyecto).

Este script se auto-verifica: despues de migrar, compara las
predicciones del modelo viejo y el nuevo sobre datos de prueba, y si la
diferencia es mayor a lo esperado por redondeo normal, se detiene sin
guardar nada. El checkpoint original nunca se modifica ni se borra.
"""

import argparse
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import TransformerLanguageModel as NewTransformerLanguageModel


# ---- Arquitectura VIEJA, tal como estaba antes de esta migracion ----
class OldHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        b, t, c = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class OldMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([OldHead(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class OldFeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)


class OldBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = OldMultiHeadAttention(n_embd, n_head, head_size, block_size)
        self.ffwd = OldFeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class OldTransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[OldBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(t, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))
        return logits, loss


def migrar_pesos(old: OldTransformerLanguageModel, new: NewTransformerLanguageModel):
    with torch.no_grad():
        new.token_embedding_table.weight.copy_(old.token_embedding_table.weight)
        new.position_embedding_table.weight.copy_(old.position_embedding_table.weight)
        new.ln_f.weight.copy_(old.ln_f.weight)
        new.ln_f.bias.copy_(old.ln_f.bias)
        new.lm_head.weight.copy_(old.lm_head.weight)
        new.lm_head.bias.copy_(old.lm_head.bias)

        for old_block, new_block in zip(old.blocks, new.blocks):
            new_block.ln1.load_state_dict(old_block.ln1.state_dict())
            new_block.ln2.load_state_dict(old_block.ln2.state_dict())
            new_block.ffwd.load_state_dict(old_block.ffwd.state_dict())

            q_w = torch.cat([h.query.weight for h in old_block.sa.heads], dim=0)
            k_w = torch.cat([h.key.weight for h in old_block.sa.heads], dim=0)
            v_w = torch.cat([h.value.weight for h in old_block.sa.heads], dim=0)
            new_block.sa.qkv.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            new_block.sa.proj.load_state_dict(old_block.sa.proj.state_dict())


def main():
    parser = argparse.ArgumentParser(description="Migra un checkpoint a la arquitectura con Flash Attention.")
    parser.add_argument("--in_checkpoint", default="atlas_lumerak/checkpoints/atlas_lumerak.pt")
    parser.add_argument("--out_checkpoint", default="atlas_lumerak/checkpoints/atlas_lumerak_flash.pt")
    args = parser.parse_args()

    device = "cpu"  # la migracion es instantanea, no hace falta GPU
    ckpt = torch.load(args.in_checkpoint, map_location=device)
    config = ckpt["config"]
    print(f"Configuracion del checkpoint: {config}")

    old_model = OldTransformerLanguageModel(**config)
    old_model.load_state_dict(ckpt["model_state_dict"])
    old_model.eval()

    new_model = NewTransformerLanguageModel(**config)
    migrar_pesos(old_model, new_model)
    new_model.eval()

    # ---- Auto-verificacion antes de guardar nada ----
    torch.manual_seed(0)
    x_test = torch.randint(0, config["vocab_size"], (2, config["block_size"]))
    y_test = torch.randint(0, config["vocab_size"], (2, config["block_size"]))

    with torch.no_grad():
        logits_old, loss_old = old_model(x_test, y_test)
        logits_new, loss_new = new_model(x_test, y_test)

    max_diff = (logits_old - logits_new).abs().max().item()
    print(f"\nPerdida modelo viejo: {loss_old.item():.6f}")
    print(f"Perdida modelo nuevo: {loss_new.item():.6f}")
    print(f"Diferencia maxima en logits: {max_diff:.2e}")

    if max_diff > 1e-3:
        print("\n¡ALERTA! La diferencia es mayor a lo esperado por redondeo normal.")
        print("NO se guarda el checkpoint nuevo. Revisar antes de continuar.")
        return

    print("\nVerificacion exitosa: el modelo nuevo se comporta identico al viejo.")
    torch.save(
        {"model_state_dict": new_model.state_dict(), "config": config},
        args.out_checkpoint,
    )
    print(f"Checkpoint migrado guardado en: {args.out_checkpoint}")
    print(f"(El original en {args.in_checkpoint} no fue modificado.)")


if __name__ == "__main__":
    main()
