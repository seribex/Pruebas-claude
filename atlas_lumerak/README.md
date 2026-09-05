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
