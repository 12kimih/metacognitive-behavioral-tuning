# MBT

Implementación oficial del artículo **"Metacognitive Behavioral Tuning of Large Language Models for Multi-Hop Question Answering"**.

**MBT** (Metacognitive Behavioral Tuning) es un marco de post-entrenamiento que inyecta una estructura metacognitiva de cinco fases en las trazas de razonamiento — *comprensión y filtrado*, *planificación*, *ejecución y monitoreo*, *autocorrección*, *verificación* — para que las conclusiones intermedias válidas sean reconocidas y preservadas en lugar de ser anuladas por una exploración continua. MBT tiene dos formulaciones:

- **MBT-S** sintetiza nuevas trazas metacognitivas desde cero.
- **MBT-R** reescribe las trazas del propio estudiante en una forma metacognitiva.

`mbt` es el código base que ejecuta MBT. Unifica (1) el despliegue de generación de datos en benchmarks de QA multi-salto / matemáticas, (2) el entrenamiento SFT en tres modos de destilación, (3) la puntuación basada en jueces (Accuracy-Efficiency Score / Reach-Redundancy Profile / Metacognitive Quality Index) — todo detrás de una única CLI de `mbt` que orquestra los backends de vLLM, OpenAI o HuggingFace.

---

## Tabla de Contenidos

- [1. Enlaces rápidos](#1-enlaces-rápidos)
- [2. Instalación](#2-instalación)
- [3. Autenticación y descarga de activos](#3-autenticación--descarga-de-activos)
- [4. Arquitectura en una pantalla](#4-arquitectura-en-una-pantalla)
- [5. Inicio rápido (prueba de humo de 10 minutos)](#5-inicio-rápido-10-minutos-prueba-de-humo)
- [6. Reproduciendo el artículo, paso a paso](#6-reproduciendo-el-artículo-paso-a-paso)
- [7. Referencia de tareas](#7-referencia-de-tareas)
- [8. Referencia del backend de la API](#8-referencia-del-backend-de-la-api)
- [9. Referencia de entrenamiento SFT](#9-referencia-de-entrenamiento-sft)
- [10. Configuraciones](#10-configuraciones)
- [11. Métricas de puntuación (AES / RRP / MQI)](#11-métricas-de-puntuación-aes--rrp--mqi)
- [12. Estructura del proyecto](#12-estructura-del-proyecto)
- [13. Envío a SLURM](#13-envío-a-slurm)
- [14. Solución de problemas](#14-solución-de-problemas)
- [15. Licencia y citación](#15-licencia--citación)

---

## 1. Enlaces rápidos

| Qué | Dónde |
|---|---|
| Organización de HF Hub | <https://huggingface.co/metacognitive-behavioral-tuning> |
| Datos de entrenamiento MBT-R | `metacognitive-behavioral-tuning/mbt-r-hotpotqa` |
| Datos de entrenamiento MBT-S | `metacognitive-behavioral-tuning/mbt-s-gpt-oss-120b` |
| Baseline Distill-R | `metacognitive-behavioral-tuning/distill-r-hotpotqa` |
| Baseline Rejection-Sampling | `metacognitive-behavioral-tuning/rollouts-hotpotqa` |
| Soluciones Gold (para MBT-R + scoring) | `metacognitive-behavioral-tuning/solutions-gpt-oss-120b` |
| Catálogo de lanzamiento de tablas del artículo (host único) | `scripts/tasks/local/*.sh` |
| Catálogo de lanzamiento de tablas del artículo (clúster SLURM) | `scripts/tasks/slurm/*.sh` |

---

## 2. Instalación

### 2.1. Prrequisitos

| Componente | Versión |
|---|---|
| Python | 3.12 |
| CUDA | 12.8 (driver 535+) |
| GPU | NVIDIA, $\ge$ 24 GB VRAM para escala 4B; se recomienda multi-GPU para 8B+ |
| Disco | $\ge$ 500 GB libres para la reproducción completa del artículo |
| Gestor de paquetes | [`uv`](https://github.com/astral-sh/uv) (reemplaza a pip / poetry) |

### 2.2. Instalar `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.3. Clonar el repositorio

```bash
git clone git@github.com:12kimih/MBT.git
cd MBT
```

### 2.4. Instalar dependencias

```bash
uv sync
```

Esto crea `.venv/` e instala el stack completo (vLLM 0.19.1, TRL 1.3+, PEFT, flash-attn 2.8, Liger Kernel, etc.) además del script de consola `mbt`.

Confirme con:

```bash
uv run mbt --help
# usage: mbt [-h] --task-name TASK_NAME ...
```

---

## 3. Autenticación y descarga de activos

### 3.1. Secretos

Copie `.env.example` a `.env` y complete las claves que necesite:

```bash
cp .env.example .env
```

```dotenv
HF_TOKEN=hf_...              # requerido: modelos/datasets restringidos de HuggingFace
OPENAI_API_KEY=sk-...        # requerido si usa backends openai.*
WANDB_API_KEY=...            # requerido para el registro de SFT (usado por sft.py)
```

`.env` es cargado automáticamente por `mbt` al iniciar, a menos que `script_config.load_dotenv=false`.

### 3.2. Inicio de sesión en una línea (recomendado)

```bash
bash scripts/setup_auth.sh
# → ejecuta:  uv run hf auth login   +   uv run wandb login
```

### 3.3. Pre-descarga de modelos y datasets

Descargue cada modelo + dataset referenciado por el pipeline de las tablas del artículo en su caché local de HF (`$HF_HOME`):

```bash
uv run python scripts/download.py
```

Puede seleccionar subconjuntos:

```bash
uv run python scripts/download.py --skip-models
uv run python scripts/download.py --skip-datasets
uv run python scripts/download.py --models "Qwen/Qwen3-4B"
```

El script reintenta 10 veces con un retroceso de 60s por repositorio; los fallos se omiten, no son fatales.

---

## 4. Arquitectura en una pantalla

```
┌──────────────────────────────────────────────────────────┐
│  mbt <task-name>                                         │
│  ─────────────────                                       │
│  task.preprocess(root_dir)  →  task_dir/requests/        │  ← prompts
│           │                                              │
│           ▼                                              │
│  api.process(task_dir)      →  task_dir/<model>/responses/│  ← inferencia
│           │                                              │
│           ▼                                              │
│  task.postprocess(api_dir)  →  task_dir/<model>/results/  │  ← columnas derivadas
└──────────────────────────────────────────────────────────┘
```

- **Tareas (Tasks)** definen *con qué datos hacer el prompt y cómo interpretar las respuestas*. Residen en `src/mbt/tasks/`. Hay 8 tareas registradas: 3 de generación de despliegues (`hotpotqa`, `musique`, `2wikimultihopqa`) y 5 de análisis (`qa.mbt_r`, `qa.evaluation`, `qa.answer_hit`, `qa.rrp_score`, `qa.mqi_score`).
- **APIs** definen *cómo ejecutar la inferencia*. Residen en `src/mbt/apis/`. Hay 2 registradas: `vllm.chat` (predeterminada, servidor local) y `openai.chat` (SDK hospedado).
- **Entrenamiento (Training)** (`src/mbt/train/sft.py`) **no** es impulsado por la CLI de `mbt` — se invoca por separado mediante `accelerate launch`.

Diseño de salida en disco (todas las tareas + APIs coinciden):

```
{root_dir}/
  <task-dir>/                # ej. hotpotqa/, mbt-r/, evaluation/
    task_config.json
    requests/                # Dataset de HF con prompts
    {model_name}/
      api_config.json
      stats.json
      logs/{timestamp}.log
      cache/response_{shard}/
      responses/             # Dataset final de HF (requests + response/valid)
      results/               # salida de task.postprocess
      results.json           # métricas agregadas por el juez (solo tareas de eval)
```

Las tareas posteriores consumen los `results/` del paso anterior directamente a través de `load_from_disk`.

---

## 5. Inicio rápido (prueba de humo de 10 minutos)

Genere 16 despliegues de HotpotQA con Qwen3-0.6B y póngalos a prueba:

```bash
# Paso 1 — despliegue (≈ 5 min en 1× A100)
uv run mbt \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "validation", "num_samples": 16}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "Qwen3-0.6B",
        "model_kwargs": {"config": "configs/vllm/qwen3-0.6b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1}
    }' \
    --script-config '{"root_dir": "output/_smoke/hotpotqa"}'

# Paso 2 — evaluación (circuito corto determinista, sin juez → solo EM/F1)
uv run mbt \
    --task-name qa.evaluation \
    --task-config '{"metrics": ["exact_match", "substring_match", "f1_score"]}' \
    --script-config '{"root_dir": "output/_smoke/hotpotqa/Qwen3-0.6B"}'

cat output/_smoke/hotpotqa/Qwen3-0.6B/results.json
```

Cuando `metrics` no contiene `"llm_as_judge"`, `qa.evaluation` ejecuta un circuito corto y escribe `results.json` directamente — no se requiere modelo juez.

---

## 6. Reproduciendo el artículo, paso a paso

La reproducción completa de las tablas del artículo está codificada como un catálogo plano de scripts de shell en `scripts/tasks/`. Dos diseños espejo:

- `scripts/tasks/local/` — ejecución directa en el host actual mediante `uv run`.
- `scripts/tasks/slurm/` — misma matriz, enviada mediante `sbatch` a un clúster SLURM.

Elija uno y ejecútelo de extremo a extremo. A continuación se muestra la variante local; reemplace `local/` por `slurm/` para el modo clúster.

### Fase 1 — Despliegues base en los tres benchmarks de QA

Para cada modelo base × {`musique`, `hotpotqa`, `2wikimultihopqa`} × {`validation`, `train`}:

```bash
bash scripts/tasks/local/rollout.sh
```

Celdas:
- 141 invocaciones en total (4 modelos × 3 datasets × 2 splits + despliegues variante).
- Salida: `output/{dataset}/{split}/{model_name}/results/` (con columnas `reasoning_trace`, `predicted_answer`).

También puede ejecutar una celda manualmente:

```bash
uv run mbt \
    --task-name musique \
    --task-config '{"dataset_split": "train"}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "Qwen3-4B",
        "model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "extra_body": {"top_k": 20}},
        "num_threads": 1024
    }' \
    --script-config '{"root_dir": "output/musique/train"}'
```

### Fase 2 — Soluciones Gold (profesor gpt-oss-120b)

```bash
bash scripts/tasks/local/solution.sh
```

Ejecuta cada tarea de QA de nivel superior con `task_config.solution=true`, lo que cambia el prompt a `SOLUTION_TEMPLATE` y almacena `solution_prompt`, `solution` (razonamiento gold) en `output/{dataset}/train/solution/gpt-oss-120b-high/results/`.

Ejemplo de una celda:

```bash
uv run mbt \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train", "solution": true}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "gpt-oss-120b-high",
        "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high"},
        "num_threads": 1024
    }' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'
```

### Fase 3 — Generar trazas sintetizadas MBT-S

```bash
bash scripts/tasks/local/mbt_s.sh
```

Igual que la Fase 2 pero con `mbt_s=true` (usa `MBT_S_TEMPLATE`). Salida:
`output/{dataset}/train/mbt-s/gpt-oss-120b-high/results/`
con la columna `synthesized_trace`. Estos son los datos de SFT para `--mode mbt-s`.

```bash
uv run mbt \
    --task-name hotpotqa \
    --task-config '{"dataset_split": "train", "mbt_s": true}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "gpt-oss-120b-high",
        "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1, "reasoning_effort": "high"},
        "num_threads": 1024
    }' \
    --script-config '{"root_dir": "output/hotpotqa/train"}'
```

### Fase 4 — Refinamiento MBT-R (reescritura de trazas del estudiante)

```bash
bash scripts/tasks/local/mbt_r.sh
```

La tarea `qa.mbt_r` consume los **despliegues anteriores del estudiante** (split train de la Fase 1) y reescribe cada `reasoning_trace` frente a la `solution` gold usando `gpt-oss-120b-high`. Salida:
`output/{dataset}/train/{student-model}/mbt-r/gpt-oss-120b-high/results/`
con columnas `refined_trace`, `trace_id`. Estos son los datos de SFT para `--mode mbt-r`.

```bash
uv run mbt \
    --task-name qa.mbt_r \
    --task-config '{"solution_config": "hotpotqa", "solution_split": "train"}' \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "gpt-oss-120b-high",
        "model_kwargs": {"config": "configs/vllm/gpt-oss-120b.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 4, "reasoning_effort": "high"},
        "num_threads": 1024
    }' \
    --script-config '{"root_dir": "output/hotpotqa/train/Qwen3-4B"}'
```

`request_kwargs.n=4` produce 4 trazas refinadas por entrada; `expand_traces` emite una fila por traza.

### Fase 5 — Entrenamiento SFT (18 celdas: 6 modos × 3 tamaños de modelo)

```bash
bash scripts/tasks/local/sft.sh
```

6 modos por Qwen3-{0.6B, 1.7B, 4B}:
- `self-distill,sft` — RS en trazas correctas generadas por el estudiante.
- `gpt-oss-distill,sft` — RS en despliegues del profesor gpt-oss-120b.
- `mbt-s,sft` — síntesis completa de MBT-S.
- `mbt-r,sft` — refinamiento MBT-R.
- `distill-r,sft` — baseline distill-R.
- `direct-r,sft` — baseline direct-R.

Tamaño de lote efectivo: `per_device=2 × grad_accum=16 × num_gpus=4 = 128`. Tasa de aprendizaje `1e-4`. 1 época. Programa coseno con `warmup_ratio=0.1`. Directorio de salida: `output/train/{model}/{mode}/sft/1e-4/128/`.

Ejemplo de una sola celda:

```bash
uv run accelerate launch \
    --config_file configs/accelerate/multi_gpu.yaml \
    --main_process_port $(shuf -i 49152-65535 -n 1) \
    src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-4B \
    --mode mbt-r \
    --wandb_tags Qwen3-4B,mbt-r,sft,1e-4,128 \
    --output_dir output/train/Qwen3-4B/mbt-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16
```

### Fase 6 — Despliegues del modelo entrenado en validación

Después de que termine la Fase 5, ejecute nuevamente rollout.sh (ya incluye las celdas de la "Sección 2" que apuntan `model_kwargs.model` a los directorios de checkpoints de SFT locales):

```bash
bash scripts/tasks/local/rollout.sh   # Las celdas de la Sección 2 reutilizan los ckpts entrenados
```

### Fase 7 — Evaluación (EM / substring / F1 / LLM-as-judge)

```bash
bash scripts/tasks/local/evaluation.sh   # 156 celdas: 52 variantes × 3 datasets
```

Modelo juez: `gemma-4-31b-it`. Celda única:

```bash
uv run mbt \
    --task-name qa.evaluation \
    --api-name vllm.chat \
    --api-config '{
        "model_name": "gemma-4-31b-it",
        "model_kwargs": {"config": "configs/vllm/gemma-4-31b-it.yaml"},
        "request_kwargs": {"temperature": 0.6, "top_p": 0.95, "n": 1}
    }' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B/mbt-r/sft/1e-4/128"}'
```

### Fase 8 — Puntuación auxiliar (métricas de la Sección 4 del artículo)

```bash
bash scripts/tasks/local/answer_hit.sh        # answer_hit
bash scripts/tasks/local/build_difficulty.sh  # pre-cálculo para mqi
bash scripts/tasks/local/rrp_score.sh         # RRP (Sección 4.2 del artículo)
bash scripts/tasks/local/mqi_score.sh         # MQI (Sección 4.3 del artículo)
```

> `mqi_score.sh` **requiere que `build_difficulty.sh` se haya ejecutado primero**, ya que este último escribe `data/sample_difficulty.csv`.

### Fase 9 — Agregación de resultados

Cada tarea escribe artefactos deterministas por celda en `output/<dataset>/<split>/<model>/<task>/<judge>/results/` (Dataset de HF) más `results.json` (métricas agregadas por el juez) para tareas de evaluación. Extraiga las columnas que necesite en un DataFrame con `datasets.load_from_disk`:

```python
from datasets import load_from_disk
ds = load_from_disk("output/musique/validation/Qwen3-4B/rrp-score/gemma-4-31b-it/results")
print(ds.column_names)
# → ['sample_id', 'reasoning_trace', 'predicted_answer',
#    'first_correct', 'redundant_fraction', 'confidence',
#    'answer_paragraph', 'redundant_paragraphs', ...]
```

Los números principales del artículo (AES, RRP, MQI) son agregaciones simples sobre estas columnas; consulte la §11 para saber qué columna corresponde a qué métrica.

---

## 7. Referencia de tareas

Cada tarea acepta una configuración JSON a través de `--task-config '{...}'`. Los valores predeterminados son constantes del módulo (ver `src/mbt/tasks/<task>.py`). Campos comunes a continuación; extras por tarea después.

### 7.1. Campos comunes de configuración de tareas

| Campo | Tipo | Predeterminado | Descripción |
|---|---|---|---|
| `dataset_name` | str | específico de tarea | Ruta del dataset en HF Hub. |
| `dataset_config` | str \| null | específico de tarea | Subconfiguración del dataset de HF. |
| `dataset_split` | str | `"validation"` | Split a cargar. |
| `num_proc` | int | `$OMP_NUM_THREADS` u 8 | Paralelismo de `dataset.map`. |
| `num_samples` | int \| null | null | Si se establece, corta a las primeras N muestras (para depuración). |
| `skip_format_columns` | bool | false | Omitir el paso de estandarización (solo tareas de QA). |

### 7.2. Tareas de QA de nivel superior — `hotpotqa`, `musique`, `2wikimultihopqa`

Banderas de modo extra (mutuamente excluyentes en la práctica):

| Bandera | Modo de salida | Plantilla de prompt |
|---|---|---|
| (ninguna) | despliegues (`reasoning_trace`, `predicted_answer`) | `PROMPT_TEMPLATE` |
| `metacognitive_prompt: true` | despliegues con prompt de sistema metacognitivo | `METACOGNITIVE_PROMPT_TEMPLATE` |
| `solution: true` | soluciones gold (`solution`, `solution_prompt`) | `SOLUTION_TEMPLATE` |
| `mbt_s: true` | traza sintetizada MBT-S (`synthesized_trace`) | `MBT_S_TEMPLATE` |

Diseño del directorio de salida:
- modo predeterminado $\to$ `{root_dir}/`
- `solution` $\to$ `{root_dir}/solution/`
- `mbt_s` $\to$ `{root_dir}/mbt-s/`

### 7.3. `qa.mbt_r` — Refinamiento MBT-R

Lee `{root_dir}/results/` (los resultados de un despliegue previo) y reescribe cada `reasoning_trace` frente a la `solution` gold. Configuración extra:

| Campo | Predeterminado |
|---|---|
| `solution_name` | `"metacognitive-behavioral-tuning/solutions-gpt-oss-120b"` |
| `solution_config` | `"hotpotqa"` |
| `solution_split` | `"train"` |

Produce N copias por solicitud (controladas por `api_config.request_kwargs.n`) con `refined_trace`, `trace_id`.

### 7.4. `qa.evaluation` — métricas deterministas + juez

| Campo | Predeterminado |
|---|---|
| `metrics` | `["exact_match", "substring_match", "f1_score", "llm_as_judge"]` |

Si `metrics` excluye `"llm_as_judge"`, el preprocesamiento ejecuta un circuito corto — no se necesita API, ni se requiere `--api-name`.

### 7.5. `qa.answer_hit` — juicio de derivación de respuesta

Sin configuración extra más allá de los campos comunes. El postprocesamiento analiza la respuesta del juez `== "YES"` $\to$ `answer_hit=1.0`.

### 7.6. `qa.rrp_score` — RRP (Reach-Redundancy Profile)

Juez de regulación basado en marcadores. Sin configuración extra. Columnas de salida: `first_correct`, `redundant_fraction`, `confidence`, `redundant_paragraphs`, `answer_paragraph`.

### 7.7. `qa.mqi_score` — MQI (Metacognitive Quality Index consciente de la longitud)

| Campo | Predeterminado |
|---|---|
| `difficulty_csv` | `"data/sample_difficulty.csv"` |
| `default_tier` | `"medium"` |

Requiere que `scripts/build_difficulty.py` se haya ejecutado primero (escribe el CSV de dificultad). Columnas de salida: `l_obs`, `phases`, `confidence`.

---

## 8. Referencia del backend de la API

Pase mediante `--api-name "<name>"` y `--api-config '{...}'`.

| Nombre registrado | Transporte | Ideal para |
|---|---|---|
| `vllm.chat` | `vllm serve` local + HTTP compatible con OpenAI | **predeterminado para ejecuciones de tablas del artículo** |
| `openai.chat` | SDK hospedado `chat.completions.create` | modelos cerrados hospedados (GPT-4o, gemini, etc.) |

### 8.1. `vllm.chat` — referencia completa de api_config

| Clave | Tipo | Predeterminado | Descripción |
|---|---|---|---|
| `model_name` (**requerido**) | str | — | Subdirectorio de salida bajo `{task_dir}`. |
| `model_kwargs` (**requerido**) | dict | — | Se reenvía a `vllm serve`. Debe incluir `config: <ruta-al-yaml>`; las claves explícitas y los seeds del yaml anulan. |
| `request_kwargs` | dict | `{}` | Se reenvía a `client.chat.completions.create`. Comunes: `temperature`, `top_p`, `n`, `max_completion_tokens`, `extra_body`. |
| `num_threads` | int | 1 | Tamaño del pool de trabajadores para solicitudes concurrentes del cliente. |
| `num_proc` | int | `$OMP_NUM_THREADS` u 8 | Paralelismo de mapa del dataset. |
| `max_retries` | int | 0 | Reintentos cuando `finish_reason ∈ retry_on`. |
| `retry_on` | list[str] | `["length", "content_filter"]` | Valores desencadenantes. |
| `log_ratio` | float | 0.01 | Frecuencia de registro de progreso como proporción del total. |
| `cache_ratio` | float | 0.1 | Frecuencia de vaciado de caché. |
| `sample_ratio` | float | 0.1 | Frecuencia de registro de muestras por shard. |
| `client_timeout` | int (seg) | 300 | Tiempo de espera de solicitud HTTPX de OpenAI. |
| `client_max_retries` | int | 20 | Reintentos de red del cliente de OpenAI. |
| `health_check_timeout` | int (seg) | 3600 | Presupuesto de sondeo de `vllm serve /health`. |
| `dry_run` | bool | false | Omitir arranque del servidor + bucle de trabajadores. |
| `seed` | int | 42 | Se reenvía a request_kwargs si no está establecido. |

El comando `vllm serve` se construye a partir de `model_kwargs` mapeando cada kv $\to$ bandera de CLI (ej. `tensor_parallel_size: 4` $\to$ `--tensor-parallel-size 4`). Los archivos yaml en `configs/vllm/<model>.yaml` proporcionan ajustes preestablecidos.

### 8.2. Ejemplo: OpenAI hospedado

```bash
uv run mbt \
    --task-name qa.evaluation \
    --api-name openai.chat \
    --api-config '{
        "model_name": "gpt-4o-judge",
        "request_kwargs": {"model": "gpt-4o-mini", "temperature": 0.0, "max_completion_tokens": 1024}
    }' \
    --script-config '{"root_dir": "output/musique/validation/Qwen3-4B"}'
```

---

## 9. Referencia de entrenamiento SFT

 Controlador: `src/mbt/train/sft.py`. Lanzado mediante `accelerate launch`, **no** a través de la CLI de `mbt`. Lee `configs/sft.yaml` para los valores predeterminados y acepta anulaciones por CLI.

### 9.1. Banderas críticas de CLI

| Bandera | Predeterminado | Descripción |
|---|---|---|
| `--config` | — | Ruta al yaml base de SFT (use `configs/sft.yaml`). |
| `--mode` | `mbt-r` | Uno de `distill` \| `mbt-s` \| `mbt-r`. Selecciona qué columna se convierte en el objetivo de completado. |
| `--model_name_or_path` | `Qwen/Qwen3-4B` | Modelo base. |
| `--dataset_name` | `metacognitive-behavioral-tuning/mbt-r-hotpotqa` | Dataset de entrenamiento en HF Hub. |
| `--dataset_config` | `Qwen3-4B` | Subconfiguración (porción del dataset por modelo). |
| `--output_dir` | predeterminado yaml | Donde van los checkpoints. |
| `--learning_rate` | `1e-4` | LR pico (programa coseno). |
| `--per_device_train_batch_size` | 2 | Lote por GPU. |
| `--gradient_accumulation_steps` | 16 | Lote efectivo = pdb × gas × n_gpus. |
| `--num_train_epochs` | 1 | |
| `--max_length` | 32768 | Límite de longitud de secuencia tokenizada. |
| `--use_peft` | false | Establecer en true para LoRA. Siguen `--lora_r`, `--lora_alpha`, `--lora_target_modules`, etc. |
| `--num_rollouts` | null | Filtrar dataset a `rollout_id <= N`. |
| `--num_traces` | null | Filtrar dataset a `trace_id <= N`. |
| `--wandb_project` | `mbt` | Nombre del proyecto W&B. |
| `--wandb_run_group` | null | Grupo de ejecuciones de W&B. |
| `--wandb_tags` | null | Etiquetas de W&B separadas por comas. |
| `--train_seed` | 42 | Seed de RNG para el entrenamiento. |

### 9.2. Modo $\to$ objetivo de completado

| Modo | Completado construido a partir de |
|---|---|
| `distill` | `example["response"]` (despliegue bruto). |
| `mbt-s` | `<think>\n{synthesized_trace}\n</think>\n\n<answer>{answer}</answer>` |
| `mbt-r` | `<think>\n{refined_trace}\n</think>\n\n<answer>{answer}</answer>` |

`tokenize` produce una `completion_mask` para que `completion_only_loss: true` en `configs/sft.yaml` aplique la pérdida solo a los tokens del asistente.

### 9.3. Ejemplo: ajuste fino completo de Qwen3-4B en MBT-R

```bash
uv run accelerate launch \
    --config_file configs/accelerate/multi_gpu.yaml \
    --main_process_port $(shuf -i 49152-65535 -n 1) \
    src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-4B \
    --mode mbt-r \
    --output_dir output/train/Qwen3-4B/mbt-r/sft/1e-4/128 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --wandb_tags Qwen3-4B,mbt-r,sft,1e-4,128
```

### 9.4. Ejemplo: ajuste fino LoRA (recursos de memoria limitados)

```bash
uv run accelerate launch \
    --config_file configs/accelerate/fsdp_qlora.yaml \
    src/mbt/train/sft.py \
    --config configs/sft.yaml \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name metacognitive-behavioral-tuning/mbt-r-hotpotqa \
    --dataset_config Qwen3-4B \
    --mode mbt-r \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_target_modules all-linear \
    --output_dir output/train/Qwen3-4B/mbt-r/lora/1e-4
```

---

## 10. Configuraciones

### 10.1. `configs/sft.yaml`

Tres dataclasses de TRL aplanadas en un único espacio de nombres YAML. Predeterminados clave:

- `completion_only_loss: true` (pérdida solo en tokens del asistente mediante `completion_mask`).
- `dataset_kwargs.skip_prepare_dataset: true` (el paso de tokenización pre-construye los input_ids).
- `attn_implementation: flash_attention_2`.
- `use_liger_kernel: true`.
- `gradient_checkpointing: true` con `use_reentrant: false`.
- `bf16: true`, `tf32: true`, `optim: adamw_torch_fused`.

### 10.2. `configs/accelerate/*.yaml`

Elija uno con `--config_file`:

| Archivo | Distribución | Uso para |
|---|---|---|
| `single_gpu.yaml` | `NO` (1 GPU) | depuración |
| `multi_gpu.yaml` | DDP, 4 GPUs | **predeterminado** para SFT |
| `fsdp.yaml` | FSDP `FULL_SHARD` + `TRANSFORMER_BASED_WRAP` | modelos muy grandes |
| `fsdp_qlora.yaml` | FSDP + QLoRA de 4 bits | modelo grande en VRAM pequeña |

### 10.3. `configs/vllm/*.yaml`

Ajustes preestablecidos de banderas de `vllm serve` por modelo. Se referencian mediante `model_kwargs.config`. Incluye presets para `qwen3-{0.6b,1.7b,4b,8b,14b,32b}.yaml`, `gpt-oss-{20b,120b}.yaml`, `gemma-4-31b-it.yaml`, `llama-4-scout-fp8.yaml`, además de varias variantes de Qwen3.5 / Qwen3.6 / Nemotron / Mistral / DeepSeek.

Ejemplo (`configs/vllm/qwen3-4b.yaml`):

```yaml
model: Qwen/Qwen3-4B
max_model_len: 32768
gpu-memory-utilization: 0.9
tensor-parallel-size: 4
trust-remote-code: true
reasoning-parser: qwen3
```

Añada anulaciones por modelo mediante `model_kwargs` en el momento de la llamada:

```json
{"model_kwargs": {"config": "configs/vllm/qwen3-4b.yaml", "max_model_len": 40960, "tensor_parallel_size": 2}}
```

---

## 11. Métricas de puntuación (AES / RRP / MQI)

El artículo introduce tres métricas de calidad de trazas. Cada una se mapea a una o más tareas:

| Métrica | Artículo | Tarea(s) | Columna(s) de salida |
|---|---|---|---|
| **EM / Substring / F1** | §3 | `qa.evaluation` (circuito corto determinista) | `exact_match`, `substring_match`, `f1_score` |
| **Precisión LLM-as-judge** | §3 | `qa.evaluation` (con `llm_as_judge` en metrics) | `llm_as_judge` |
| **Tasa de acierto de respuesta** | §3 | `qa.answer_hit` | `answer_hit`, `substring_match` |
| **Accuracy-Efficiency Score (AES)** | §4.1 | computado posteriormente desde las columnas de EM + conteo de tokens | derivado del conteo de tokens + EM |
| **Reach-Redundancy Profile (RRP)** | §4.2 | `qa.rrp_score` | `first_correct`, `redundant_fraction`, `confidence`, `answer_paragraph`, `redundant_paragraphs` |
| **Metacognitive Quality Index (MQI)** | §4.3 | `qa.mqi_score` | `l_obs`, `phases`, `confidence` |

Para detalles de implementación de RRP y MQI, vea `docs/scoring_redesign_marker_variant.md`.

---

## 12. Estructura del proyecto

```
MBT/
├── src/mbt/                # Paquete central (Python 3.12, script de consola: `mbt`)
│   ├── main.py              # Orquestador del pipeline (preprocess → API → postprocess)
│   ├── registry.py          # decoradores @register_task / @register_api
│   ├── apis/                # Backends de inferencia
│   │   ├── vllm/chat.py     # `vllm serve` local + HTTP compatible con OpenAI
│   │   └── openai/chat.Lpy   # SDK de OpenAI hospedado
│   ├── tasks/               # Definiciones de tareas
│   │   ├── hotpotqa.py      # modos rollout / solution / MBT-S
│   │   ├── musique.py
│   │   ├── 2wikimultihopqa.py
│   │   └── qa/              # 5 tareas de análisis (mbt_r + evaluation + answer_hit + rrp_score + mqi_score)
│   └── train/               # Entrenador SFT (TRL)
├── configs/
│   ├── sft.yaml             # Configuración SFT de TRL (3 dataclasses planas)
│   ├── accelerate/          # Lanzadores distribuidos
│   └── vllm/                # Presets de `vllm serve` por modelo
├── scripts/
│   ├── setup_auth.sh        # inicio de sesión hf + wandb
│   ├── download.py          # Pre-descarga de modelos y datasets de HF
│   ├── build_difficulty.py  # Pre-cálculo de dificultad por muestra (entrada para MQI)
│   ├── slurm/               # puntos de entrada SBATCH *.slurm
│   └── tasks/
│       ├── local/           # Catálogo de host único (replicación de tablas del artículo)
│       └── slurm/           # Mismo catálogo, impulsado por sbatch
├── pyproject.toml           # Metadatos del proyecto + dependencias (gestionado por uv)
├── uv.lock                  # Bloqueo de dependencias fijadas
└── README.md                # ← usted está aquí
```

---

## 13. Envío a SLURM

Cada `*.sh` en `scripts/tasks/local/` tiene un gemelo en `slurm/` que envuelve cada celda como:

```
sbatch --cpus-per-task=32 --gres=gpu:4 scripts/slurm/<entrada>.slurm <args>
```

Los encabezados de `scripts/slurm/*.slurm` dejan `--partition`, `--qos` y `--time` en blanco — edítelos para su clúster antes de enviar. `OMP_NUM_THREADS` se calcula automáticamente:

```bash
((SLURM_GPUS_ON_NODE > 0)) && export OMP_NUM_THREADS=$((SLURM_CPUS_ON_NODE / SLURM_GPUS_ON_NODE))
```

Envíe la matriz de reproducción completa:

```bash
bash scripts/tasks/slurm/rollout.sh
bash scripts/tasks/slurm/solution.sh
bash scripts/tasks/slurm/mbt_s.sh
bash scripts/tasks/slurm/mbt_r.sh
bash scripts/tasks/slurm/sft.sh
bash scripts/tasks/slurm/rollout.sh    # Sección 2 — variantes entrenadas
bash scripts/tasks/slurm/evaluation.sh
bash scripts/tasks/slurm/answer_hit.sh
bash scripts/tasks/slurm/build_difficulty.sh
bash scripts/tasks/slurm/rrp_score.sh
bash scripts/tasks/slurm/mqi_score.sh
```

---

## 14. Solución de problemas

### 14.1. vLLM se cuelga al iniciar / OOMs

Revise `<api_dir>/logs/<timestamp>_server.log`. Soluciones comunes:
- Reduzca `gpu-memory-utilization` en el archivo `configs/vllm/<model>.yaml` del modelo (ej. 0.9 $\to$ 0.85).
- Reduzca `max_model_len` (ej. 32768 $\to$ 16384) para liberar la caché KV.
- Si un trabajador de vLLM huérfano está reteniendo el puerto de una ejecución fallida anterior, localícelo y mátelo manualmente: `pgrep -fu "$USER" 'vllm serve' | xargs -r kill -TERM`.

### 14.2. `ModuleNotFoundError: No module named 'mbt'`

No está en el entorno de `uv`. Use `uv run` para todos los comandos, o `source .venv/bin/activate`.

### 14.3. La tarea está registrada pero nunca se ejecuta

Inspeccione los registros — la causa más común es un error de sintaxis en cualquier otro módulo bajo `mbt.apis.*` o `mbt.tasks.*`. `recursive_import` falla silenciosamente para hermanos de un módulo roto. Ejecute:

```bash
uv run python -c "
from mbt.main import recursive_import
from mbt.registry import TASK_REGISTRY, API_REGISTRY
recursive_import('mbt.apis')
recursive_import('mbt.tasks')
print(len(TASK_REGISTRY), 'tasks;', len(API_REGISTRY), 'apis')
"
# Debería imprimir: 8 tasks; 2 apis
```

### 14.4. `/tmp` es `noexec` (algunos clústeres compartidos)

vLLM utiliza cachés de Triton + torch.inductor que por defecto van a `/tmp`. Rediríjalas a un montaje escribible+ejecutable antes de lanzar:

```bash
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TMPDIR="$HOME/.cache/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
```

### 14.5. Los modelos pre-descargados se siguen descargando

Asegúrese de que `$HF_HOME` esté exportado en el shell que lanza `mbt` (algunos clústeres reinician el entorno en `srun`). `download.py` respeta `HF_HOME`.

---

## 15. Licencia y citación

Este proyecto está licenciado bajo **Apache-2.0**. Vea [LICENSE](LICENSE).

Si utiliza este código o los checkpoints / datasets publicados, por favor cite:

```bibtex
@misc{kim2026metacognitivebehavioraltuninglarge,
      title={Metacognitive Behavioral Tuning of Large Language Models for Multi-Hop Question Answering},
      author={Ik-hwan Kim and Hyeongrok Han and Mingi Jung and Sangwon Yu and Jinseok Hong and Sang Hun Kim and Yoonyoung Choi and Sungroh Yoon},
      year={2026},
      eprint={2602.22508},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.22508},
}
```
