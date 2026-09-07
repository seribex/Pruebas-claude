"""
Bateria fija de preguntas para comparar modelos con el mismo rasero.

Juzgar "a ojo" con preguntas distintas cada vez y muestreo al azar no
sirve para decidir nada: ya nos paso una vez que sacamos una conclusion
firme de cinco respuestas sueltas y estaba equivocada. Este script hace
siempre las MISMAS preguntas, con la MISMA semilla, para que la unica
diferencia entre dos corridas sea el modelo o los ajustes de muestreo.

Uso tipico -- comparar el modelo viejo con el nuevo:
  python atlas_lumerak/probar_chat.py --checkpoint ...checkpoints_chat/atlas_lumerak_mejor.pt
"""

import argparse
import textwrap

import torch

from checkpoint_utils import cargar_modelo
from inferencia import recortar_respuesta

PREGUNTAS = [
    # Identidad
    "Hola",
    "¿Cómo te llamas?",
    "¿Quién te creó?",
    # Instrucciones simples: la respuesta DEBE hablar del tema pedido
    "Dame 3 ideas para estudiar mejor",
    "Explícame qué es la fotosíntesis",
    "Escribe una lista de 4 frutas",
    "¿Cuál es la capital de Francia?",
    "Resume en una frase qué es el agua",
    # Instruccion con formato explicito
    "Escribe un saludo corto en tono formal",
    # Deberia admitir que no sabe
    "¿Qué hora es?",
]


def main():
    p = argparse.ArgumentParser(description="Compara modelos con preguntas fijas.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--length", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_k", type=int, default=40, help="0 para desactivar")
    p.add_argument("--top_p", type=float, default=0.9, help="1.0 para desactivar")
    p.add_argument("--repeticion", type=float, default=1.15)
    p.add_argument("--semilla", type=int, default=1234)
    p.add_argument("--sin_recorte", action="store_true",
                   help="Mostrar todo lo que genera, sin cortar en la primera linea en blanco.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, config = cargar_modelo(args.checkpoint, device, args.vocab)
    print(f"Modelo: {args.checkpoint}")
    print(f"Ajustes: temperatura {args.temperature}, top_k {args.top_k}, "
          f"top_p {args.top_p}, repeticion {args.repeticion}\n")

    for i, pregunta in enumerate(PREGUNTAS):
        # Semilla por pregunta: dos corridas del mismo modelo dan el mismo
        # resultado, asi que cualquier diferencia que veas es real.
        torch.manual_seed(args.semilla + i)
        ids = tok.encode(f"Usuario: {pregunta}\nAtlas:")[-config["block_size"]:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        salida = model.generate(
            idx, max_new_tokens=args.length,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p < 1.0 else None,
            repetition_penalty=args.repeticion,
        )[0].tolist()
        respuesta = recortar_respuesta(tok.decode(salida[len(ids):]),
                                       parar_en_blanco=not args.sin_recorte)
        print(f"--- {i + 1}. {pregunta}")
        print(textwrap.indent(respuesta or "(vacio)", "    "))
        print()


if __name__ == "__main__":
    main()
