# 🧠 Metacognitive Behavioral Tuning (MBT)

Method code, prompts and data samples for **Distilling Metacognitive Behaviors through
Agentic Editing**. MBT adds metacognitive behaviors to teacher
reasoning traces for subsequent supervised fine-tuning.

[Quickstart](#-quickstart) | [Methods](#methods) | [Data](#-data) | [Models](#-models) |
[Results](#-results-from-the-paper) | [Usage guide](docs/usage.md) | [Citation](#-citation)

## 🚀 Quickstart

From this directory:

```bash
pip install -e ".[test]"
pytest -q
export DEEPSEEK_API_KEY=...

mbt generate --input examples/problems.jsonl --output out/unedited.jsonl --endpoint deepseek
mbt rewrite --input examples/traces.jsonl --output out/mbt-r.jsonl --endpoint deepseek
mbt insert --input examples/traces.jsonl --output out/mbt-i.jsonl --endpoint deepseek

docker build -t mbt-editor:latest -f docker/Dockerfile docker
mbt agentic --input examples/traces.jsonl --output out/mbt-a.jsonl --sandbox docker \
    --artifacts out/mbt-a-artifacts
mbt sft --input out/mbt-a.jsonl --output out/sft-mbt-a.jsonl
```

Model commands require API access and save completed rows for resume. Keep each output
with its `.manifest.json`. Resume with the same inputs, code and configuration, or use a
new output path. See the [usage guide](docs/usage.md) for input fields, error recovery,
endpoint settings, E2B setup, annotation and prompts. Run `mbt COMMAND --help` for options.
For E2B support, install `pip install -e ".[e2b]"`.

## Methods

The editor receives the problem, reasoning and response, without the reference answer.
The methods share seven behavior instructions: subgoal setting, logical check, substantive
verification, targeted backtracking, resuming abandoned paths, self-correction and
sufficiency judgment.

| Condition | Operation |
|---|---|
| Unedited | Generate the teacher reasoning and response |
| MBT-R | Rewrite the trace in one response |
| MBT-I | Return insertions at paragraph markers |
| MBT-A | Edit `trajectory.md` in a Docker or E2B sandbox, recording changes in Git |

Prompts are included in `src/mbt/prompts/`. Offline tests check the editing prompt hashes.
The appendix grading prompts are included as [reference texts](src/mbt/prompts/grading/README.md).
Training and evaluation code are excluded.

## 📦 Data

The data package contains 1,000 Nemotron-Math-v2 problems, one trace per problem for
Unedited, MBT-R, MBT-I and MBT-A, and one SFT config per condition. Failed MBT-R edits remain in the
trace file with their status; only successful edits enter its SFT config.

| Condition | Trace rows | SFT rows | SFT tokens (paper Table D1) |
|---|---:|---:|---:|
| Unedited | 1,000 | 995 | 22,821,876 |
| MBT-R | 1,000 | 773 | 7,557,835 |
| MBT-I | 1,000 | 995 | 23,483,410 |
| MBT-A | 1,000 | 995 | 24,432,120 |

SFT rows fit the 131,072-token window under the Qwen3.5-4B chat template.
The separate data package includes the dataset card, field descriptions, attribution
and JSONL files. The dataset is on Hugging Face at
[`12kimih/mbt-traces`](https://huggingface.co/datasets/12kimih/mbt-traces).
The included examples can be used without downloading it.

## 🤗 Models

Qwen3.5-4B-Base fine-tuned on each condition, in bf16:

| Condition | Model |
|---|---|
| Unedited SFT | [`12kimih/Qwen3.5-4B-Unedited-SFT`](https://huggingface.co/12kimih/Qwen3.5-4B-Unedited-SFT) |
| MBT-R | [`12kimih/Qwen3.5-4B-MBT-R`](https://huggingface.co/12kimih/Qwen3.5-4B-MBT-R) |
| MBT-I | [`12kimih/Qwen3.5-4B-MBT-I`](https://huggingface.co/12kimih/Qwen3.5-4B-MBT-I) |
| MBT-A | [`12kimih/Qwen3.5-4B-MBT-A`](https://huggingface.co/12kimih/Qwen3.5-4B-MBT-A) |

## 📊 Results (from the paper)

Qwen3.5-4B-Base trained for 15 epochs. Math is avg@8 over the 153 problems of AIME 2025 and 2026
and HMMT February 2025, February 2026 and November 2025, graded by an LLM judge.

**Competition math, avg@8 % (Table 2)**

| Condition | AIME 25 | AIME 26 | HMMT Feb 25 | HMMT Feb 26 | HMMT Nov 25 | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| Base model | 84.2 | 84.6 | 78.3 | 69.3 | 73.3 | 77.8 |
| Unedited SFT | 85.4 | 84.6 | 83.8 | 78.4 | 83.8 | 83.1 |
| MBT-R | 70.0 | 74.6 | 82.5 | 65.5 | 77.5 | 73.9 |
| MBT-I | 82.9 | 87.5 | 85.0 | 78.8 | 85.4 | 83.8 |
| MBT-A | 85.0 | 90.0 | 85.0 | 79.9 | 87.5 | 85.4 |

**Untrained domains, % (Table 3)**: avg@8 for GPQA Diamond and LiveCodeBench v6, one sample for the
multi-hop QA sets.

| Condition | GPQA | LCB | MuSiQue | HotpotQA | 2Wiki |
|---|---:|---:|---:|---:|---:|
| Base model | 72.0 | 47.1 | 63.0 | 88.8 | 83.2 |
| Unedited SFT | 77.5 | 47.5 | 74.3 | 91.3 | 85.4 |
| MBT-R | 77.5 | 43.5 | 69.3 | 90.1 | 83.7 |
| MBT-I | 79.2 | 48.0 | 74.8 | 91.3 | 85.7 |
| MBT-A | 78.1 | 50.3 | 74.6 | 91.4 | 86.0 |

## Licenses

Code and original prompts: [Apache-2.0](LICENSE). Third-party prompt licenses are
listed in [NOTICE](NOTICE). Data: CC BY-SA 4.0, with upstream
attribution in the data package. See [NOTICE](NOTICE) for third-party notices.

## 📚 Citation

```bibtex
@misc{mbt2026,
  title  = {Distilling Metacognitive Behaviors through Agentic Editing},
  author = {Kim, Ik-hwan and Han, Hyeongrok and Jung, Mingi and Yu, Sangwon and Hong, Jinseok and Kim, Sang Hun and Choi, Yoonyoung and Yoon, Sungroh},
  year   = {2026},
  note   = {Under review}
}
```
