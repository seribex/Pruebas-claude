"""
Carga de modelos ya entrenados, compartida por generate.py y chat.py.

Un checkpoint por si solo no alcanza: hacen falta los pesos Y el
tokenizador con el que fue entrenado. Si se mezclan (pesos de un modelo
con el tokenizador de otro), el modelo genera basura sin dar ningun
error, porque los numeros significan otra cosa. Por eso cada checkpoint
guarda que tokenizador le corresponde.
"""

import os

import torch

from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer
from model import TransformerLanguageModel


def cargar_modelo(ruta_checkpoint: str, device: str, ruta_tokenizador: str | None = None):
    ckpt = torch.load(ruta_checkpoint, map_location=device)

    # Los checkpoints viejos, anteriores a BPE, no tienen este campo:
    # todos ellos usaban tokenizador de caracteres.
    info = ckpt.get("tokenizador", {"tipo": "char", "archivo": "vocab.json"})

    ruta_tok = ruta_tokenizador or os.path.join(
        os.path.dirname(ruta_checkpoint) or ".", info["archivo"]
    )
    if info["tipo"] == "bpe":
        tokenizer = BPETokenizer.load(ruta_tok)
    else:
        tokenizer = CharTokenizer.load(ruta_tok)

    model = TransformerLanguageModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer, ckpt["config"]
