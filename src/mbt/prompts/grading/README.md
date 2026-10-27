# Reference grading prompts

These text assets provide the grading prompts identified in the paper appendix.

| File | Use | Source |
| --- | --- | --- |
| `math.txt` | Competition math | NVIDIA NeMo-Skills, `judge/math.yaml` |
| `gpqa.txt` | GPQA Diamond | HLE, commit `22ed3074`, `JUDGE_PROMPT` |
| `qa.txt` | MuSiQue, HotpotQA, 2WikiMultihopQA | SimpleQA Verified, inspect_evals commit `dbd3dd25` |
| `qa_instructions.txt` | The `{instructions}` slot in `qa.txt` | The same SimpleQA source |

The templates are user-message text. Replace their parameter markers with the
corresponding problem and answers. For `qa.txt`, the `{instructions}` marker takes
the complete text of `qa_instructions.txt`. This directory contains reference
texts only; the package does not include an evaluation or grading runner.

`sources.json` records source locations, versions, file hashes and prompt hashes.
The NeMo text uses `<<PROBLEM>>`, `<<CANDIDATE>>` and `<<REFERENCE>>` markers;
its escaped format-string braces are decoded to literal LaTeX braces. These are
template-encoding changes, recorded explicitly in `sources.json`. The HLE and
SimpleQA texts match the cited source constants.

## Attribution and licenses

- The NeMo-Skills template uses [Apache-2.0](../../../../LICENSE). Its upstream
  prompt states that it is adapted from OpenAI simple-evals.
- HLE: [MIT license](licenses/hle-MIT.txt).
- inspect_evals: [MIT license](licenses/inspect-evals-MIT.txt).
- OpenAI simple-evals: [MIT license](licenses/simple-evals-MIT.txt).

Third-party prompt texts retain their upstream licenses and attribution.
