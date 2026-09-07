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


def ampliar_vocabulario(estado: dict, config: dict, nuevo_tamano: int):
    """Agranda las tablas de entrada y salida de un modelo ya entrenado.

    Hace falta para anadirle a Atlas un simbolo que no existia cuando se
    entreno (el fin de turno). Las tablas que dependen del vocabulario son
    tres: la de entrada (numero -> vector), la de salida (vector -> puntaje
    de cada simbolo) y su sesgo.

    Las filas nuevas se inicializan con el PROMEDIO de las que ya hay, no al
    azar: un vector aleatorio seria un valor extremo en un espacio donde
    todos los demas ya estan colocados, y desestabilizaria las primeras
    actualizaciones. El promedio arranca en el centro de lo conocido.

    Los simbolos viejos conservan su numero y su vector, asi que el modelo
    sigue significando exactamente lo mismo para todo el texto anterior:
    verificado numericamente (diferencia 0.0) en las pruebas.
    """
    viejo = config["vocab_size"]
    if nuevo_tamano == viejo:
        return estado, config
    if nuevo_tamano < viejo:
        raise ValueError(f"No se puede encoger el vocabulario ({viejo} -> {nuevo_tamano}): "
                         "los simbolos sobrantes perderian su significado.")
    if config.get("tie_weights"):
        raise ValueError("Ampliar el vocabulario con pesos atados no esta soportado: "
                         "las dos tablas son el mismo tensor y se pisarian.")

    estado = dict(estado)
    faltan = nuevo_tamano - viejo
    for clave in ("token_embedding_table.weight", "lm_head.weight"):
        w = estado[clave]
        estado[clave] = torch.cat([w, w.mean(dim=0, keepdim=True).repeat(faltan, 1)], dim=0)
    if "lm_head.bias" in estado:
        b = estado["lm_head.bias"]
        estado["lm_head.bias"] = torch.cat([b, b.mean().repeat(faltan)])

    config = dict(config)
    config["vocab_size"] = nuevo_tamano
    return estado, config


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
