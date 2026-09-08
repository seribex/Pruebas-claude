"""
Un chat en la terminal para conversar con Atlas Lumerak.

Usa el mismo formato de turnos ("Usuario: ...\nAtlas: ...\n\n") con el que
se preparan los datos de conversacion (ver prepare_oasst2.py), para que el
modelo reconozca cuando le toca responder.

Ademas, guarda cada intercambio en un archivo con tu calificacion de si la
respuesta fue util. Esas conversaciones marcadas como utiles son la materia
prima para seguir mejorando a Atlas mas adelante.
"""

import argparse
import json
import time

import torch

from checkpoint_utils import cargar_modelo
from bpe_tokenizer import FIN_TURNO
from inferencia import id_fin_de_turno, recortar_respuesta


def main():
    parser = argparse.ArgumentParser(description="Chatea con un Atlas Lumerak ya entrenado.")
    parser.add_argument("--checkpoint", default="atlas_lumerak/checkpoints/atlas_lumerak.pt")
    parser.add_argument("--vocab", default=None,
                        help="Tokenizador (por defecto: el que indique el checkpoint, en su misma carpeta).")
    parser.add_argument("--response_length", type=int, default=200)
    parser.add_argument("--log", default="atlas_lumerak/data/chat_log.jsonl")
    parser.add_argument("--correcciones", default="atlas_lumerak/data/correcciones_tuyas.jsonl",
                        help="Donde se guardan tus correcciones. Cada una es una leccion que "
                             "ningun corpus del mundo tiene: nadie mas sabe como quieres que "
                             "conteste Atlas.")
    parser.add_argument("--sin_recorte", action="store_true",
                        help="Mostrar todo lo que genera, sin cortar en la primera linea en blanco.")
    # Valores mas "frios" que los de generate.py a proposito. Conversando,
    # un solo token desafortunado descarrila la respuesta entera, y en un
    # modelo de este tamano eso pasa seguido. Comprobado sobre la bateria de
    # preguntas: eligiendo SIEMPRE la palabra mas probable (sin azar), Atlas
    # contesta "Me llamo Atlas Lumerak" y "La capital de Francia es Paris";
    # con temperatura 0.7 contestaba "Me alegra que te llames". Sabia la
    # respuesta -- el azar se la arruinaba. Al escribir texto libre esa
    # variedad se agradece; al responder una pregunta, estorba.
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=40, help="0 para desactivar")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Se queda con los pocos candidatos que junten esta probabilidad. "
                             "1.0 para desactivar.")
    parser.add_argument("--repeticion", type=float, default=1.15,
                        help="Mayor a 1.0 castiga lo que acaba de decir, para frenar bucles.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = cargar_modelo(args.checkpoint, device, args.vocab)
    block_size = config["block_size"]
    id_fin = id_fin_de_turno(tokenizer)
    if id_fin is not None:
        print("(este Atlas sabe decir cuando ha terminado de hablar)")

    print("=== Atlas Lumerak ===")
    print(f"(memoria de hasta {block_size} tokens -- escribe 'salir' para terminar)\n")

    # Cada elemento es un turno completo ("Usuario: ...\nAtlas: ...\n\n"),
    # para poder descartarlos enteros cuando ya no quepan.
    turnos: list[str] = []
    while True:
        try:
            entrada = input("Tu: ")
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if entrada.strip().lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        # Si alguien escribe el simbolo de fin a mano, se quita: dejarlo
        # pasar meteria un final de turno falso en medio del contexto.
        entrada = entrada.replace(FIN_TURNO, "")
        turnos.append(f"Usuario: {entrada}\nAtlas:")

        # El modelo solo alcanza a ver block_size tokens. Se descartan los
        # turnos MAS VIEJOS enteros hasta que quepa, en vez de cortar por
        # el token numero 512: un corte a ciegas puede dejar media palabra
        # o media pregunta al principio, y el modelo empieza a leer basura.
        while True:
            contexto = "".join(turnos)
            ids = tokenizer.encode(contexto)
            if len(ids) <= block_size or len(turnos) == 1:
                break
            turnos.pop(0)
        ids = ids[-block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        salida = model.generate(
            idx,
            max_new_tokens=args.response_length,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p < 1.0 else None,
            repetition_penalty=args.repeticion,
            stop_id=id_fin,
        )[0].tolist()

        crudo = tokenizer.decode(salida[len(ids):])
        respuesta = recortar_respuesta(crudo, parar_en_blanco=not args.sin_recorte)
        print(f"Atlas: {respuesta}\n")

        # Se rearma el turno en el MISMO formato con el que se entreno: la
        # respuesta, el simbolo de fin y un salto de linea.
        cierre = f"{FIN_TURNO}\n" if id_fin is not None else "\n\n"
        turnos[-1] += f" {respuesta}{cierre}"

        util = input("¿Fue util esta respuesta? (s/n, Enter para omitir): ").strip().lower()

        # Si estuvo mal, se le puede ensenar la respuesta correcta. Eso
        # convierte cada equivocacion en material de entrenamiento: no solo
        # "esto estuvo mal", sino "esto estuvo mal Y asi se hace bien", que
        # es lo unico con lo que se puede corregir de verdad.
        correccion = ""
        if util == "n":
            correccion = input("¿Cuál habría sido la respuesta correcta? "
                               "(Enter para omitir): ").strip()
            if correccion:
                with open(args.correcciones, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "pregunta": entrada, "mala": respuesta, "buena": correccion,
                        "motivo": "corregido por ti", "tipo": "humano",
                        "checkpoint": args.checkpoint,
                        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, ensure_ascii=False) + "\n")
                # La conversacion sigue con la respuesta BUENA en el contexto,
                # no con la equivocada: asi el resto del dialogo no arrastra
                # el error, igual que cuando corriges a alguien de palabra.
                respuesta = correccion
                print(f"Atlas (corregido): {respuesta}\n")

        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "usuario": entrada,
                "atlas": respuesta,
                "util": util == "s" if util in ("s", "n") else None,
                "correccion": correccion or None,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
