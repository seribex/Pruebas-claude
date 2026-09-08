# Atlas Lumerak

Un modelo de lenguaje tipo Transformer, construido desde cero (sin usar la
API de ningun otro modelo), paso a paso.

## Estructura

- `tokenizer.py` — convierte texto en numeros y de vuelta (Paso 1)
- `bigram_model.py` — el modelo mas simple posible, referencia historica (Paso 2)
- `attention_model.py` — primera version con atencion multi-cabeza (Paso 3)
- `transformer_model.py` — Transformer completo (bloques + feed-forward) a escala de prueba (Paso 4)
- `model.py` — la misma arquitectura del Transformer, reutilizable y configurable (usada por `train.py`)
- `train.py` — entrena el modelo sobre un archivo de texto real
- `generate.py` — genera texto con un modelo ya entrenado, sin reentrenar
- `bpe_tokenizer.py` — tokenizador BPE por fragmentos de palabra (Paso 7)
- `prepare_tokens.py` — pre-codifica el corpus completo a numeros, una sola vez (Paso 8)
- `prepare_sft.py` — arma el corpus de instrucciones **con mascara de perdida** (Paso 10)
- `identidad.py` — quien es Atlas Lumerak, en ejemplos de conversacion
- `train_sft.py` — fine-tuning conversacional (Paso 11)
- `chat.py` — conversar con Atlas en la terminal
- `probar_chat.py` — bateria fija de preguntas para comparar modelos con el mismo rasero
- `data/quijote.txt` — corpus de entrenamiento: "El ingenioso hidalgo don Quijote de la Mancha" (Cervantes, dominio publico, via Project Gutenberg)

## Preparar tu computadora (Windows, con GPU NVIDIA)

1. Clona el repositorio y entra a la rama de trabajo:
   ```
   git clone https://github.com/seribex/Pruebas-claude.git
   cd Pruebas-claude
   git checkout claude/hola-zle5jo
   ```

2. Instala PyTorch **con soporte de GPU** (el `pip install torch` normal instala
   solo la version de CPU, mucho mas lenta):
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   ```

3. Verifica que tu GPU es reconocida:
   ```
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```
   Deberia imprimir `True` y el nombre de tu tarjeta (ej. `NVIDIA GeForce RTX 4060 Ti`).

## Entrenar

Desde la carpeta raiz del repositorio (`Pruebas-claude`):

```
python atlas_lumerak/train.py
```

Parametros por defecto: modelo de ~128 dimensiones, 4 capas, 4 cabezas de
atencion, 5000 pasos. Se pueden ajustar, por ejemplo para un modelo mas
grande:

```
python atlas_lumerak/train.py --n_embd 256 --n_layer 6 --n_head 8 --steps 10000
```

El resultado (pesos entrenados + vocabulario) se guarda en
`atlas_lumerak/checkpoints/` (esa carpeta no se sube a git — son archivos
generados, no codigo fuente).

## Generar texto

```
python atlas_lumerak/generate.py --prompt "En un lugar de la Mancha"
```

## Ensenarle a conversar (fine-tuning de instrucciones)

Despues del pre-entrenamiento, Atlas escribe espanol fluido pero no responde
lo que se le pregunta. Eso se arregla en una segunda etapa, con ejemplos de
pregunta y respuesta.

### 1. Armar el corpus de instrucciones

```
python atlas_lumerak/prepare_sft.py --bpe atlas_lumerak/data/atlas_bpe.json \
    --fuentes oasst2 dolly identidad --block_size 512
```

Fuentes disponibles en `--fuentes`:

| Fuente | Ejemplos | Licencia | Quien lo escribio |
|---|---|---|---|
| `oasst2` | 13,740 dialogos (18,696 turnos) | Apache 2.0 | voluntarios humanos |
| `dolly` | 15,015 | CC-BY-SA 3.0 | empleados de Databricks, traducido |
| `identidad` | 100 (repetidos) | propio | escrito para este proyecto |
| `alpaca` | 51,942 | CC-BY 4.0 | sintetico (text-davinci-003) |
| `openhermes` | hasta 1,000,000 | Apache 2.0 | sintetico (GPT-4), traducido al espanol |

**El tokenizador tiene que ser el mismo del pre-entrenamiento.** Con otro, los
pesos ya entrenados no significarian nada.

Genera tres archivos: `*_tokens.bin`, `*_mask.bin` y `*_meta.json`. La mascara
es la pieza clave: marca con 1 los tokens que dice Atlas y con 0 los de la
pregunta, para que el entrenamiento **solo aprenda a responder** y no a
inventarse preguntas.

### 2. Entrenar

```
python atlas_lumerak/train_sft.py \
    --sft atlas_lumerak/data/atlas_sft \
    --checkpoint atlas_lumerak/checkpoints/atlas_lumerak.pt \
    --bpe atlas_lumerak/data/atlas_bpe.json \
    --tokens_pre atlas_lumerak/data/atlas_tokens.bin
```

`--tokens_pre` mezcla un 20% de lotes del corpus original de Wikipedia: sin
eso, el modelo se especializa tanto en conversar que empieza a olvidar lo que
sabia.

Guarda dos modelos: `atlas_lumerak_mejor.pt` (el de mejor validacion, que es
el que hay que usar) y `atlas_lumerak.pt` (el ultimo paso, normalmente peor
por sobreajuste). Si se corta la luz, `--continuar` retoma donde quedo.

### 3. Comparar antes y despues

```
python atlas_lumerak/probar_chat.py --checkpoint atlas_lumerak/checkpoints_chat/atlas_lumerak_mejor.pt
```

Hace siempre las mismas preguntas con la misma semilla, para que la unica
diferencia entre dos corridas sea el modelo.

## Conversar

```
python atlas_lumerak/chat.py --checkpoint atlas_lumerak/checkpoints_chat/atlas_lumerak_mejor.pt
```

Ajustes de muestreo por defecto: temperatura 0.7, `top_k` 40, `top_p` 0.9 y
penalizacion de repeticion 1.15. Son mas "frios" que los de `generate.py` a
proposito: conversando, un solo token desafortunado descarrila la respuesta
entera, y en un modelo de este tamano eso pasa seguido.
