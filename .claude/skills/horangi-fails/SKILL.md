---
name: horangi-fails
description: Deep-dive error pattern analysis for a single (model, benchmark) pair using Weave traces. Surfaces how/why the model is getting answers wrong — answer bias, format violations, language mixing, and 3-5 representative failure samples. Invoke when the user asks to analyze wrong answers / failure patterns for a specific benchmark (e.g. "analyze errors in <bench>", "<model>의 <benchmark> 오답 패턴 분석", "틀린 문제 경향"). Commonly invoked as a follow-up to `horangi-analyze` when that skill flags a weak category.
---

# horangi-fails

Produce a targeted failure analysis for **one** (model config, benchmark) pair. Output goes to chat as markdown.

## Inputs

- `<config>`: YAML filename in `configs/models/` (without `.yaml`).
- `<benchmark>`: benchmark name from `ALL_BENCHMARKS` in `run_eval.py` (e.g. `kmmlu_pro`, `ko_arc_agi`, `bfcl`).
- W&B `entity/project/run_id` — **always ask** the user if not provided; do not default to any specific project.

If any input is missing or ambiguous, ask — don't guess.

## Workflow

### 1. Locate sample data

**Check cached data first**: look for `analyses/data/<config>__<benchmark>.json`. If it exists, load from it.

**Otherwise, fetch from Weave** via `analyses/retrieve_eval.py`:

```python
import sys; sys.path.insert(0, '.')
from analyses.retrieve_eval import find_eval_call, get_eval_samples

result = find_eval_call("entity/project", "run_id", "benchmark_name")
if result:
    samples = get_eval_samples("entity/project", result["call_id"])
```

After retrieving, **save the raw sample data** to `analyses/data/<config>__<benchmark>.json` for reuse:
```python
import json
with open(f"analyses/data/{config}__{benchmark}.json", "w") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
```

Each sample dict has: `input`, `completion`, `parsed_answer`, `is_correct`, `scores`, `display_name`.

### 2. Compute pass/fail

A sample is correct if `s["is_correct"] == True`. Everything else is wrong.

### 3. Pattern signals

For the wrong-answer subset, compute:

**(a) Target distribution skew**
Group wrong samples by `s["parsed_answer"]`. If one label dominates (≥30% of wrong), note "모델이 <label>을 체계적으로 선호".

**(b) Input length bucket**
Split samples into length quartiles by `len(s["input"])`. Compare wrong-rate per bucket. Call out if the tail (longest 25%) has 2× the wrong-rate of the shortest.

**(c) Output format violations**
Count wrong samples where `s["parsed_answer"] == ""` (no parseable answer). For multiple-choice benchmarks: count outputs that don't end with "정답: X" format.

**(d) Language mixing**
Count wrong samples whose completion contains significant non-Korean tokens (e.g. long English chunks). Heuristic: `re.search(r'[a-zA-Z]{30,}', completion)`.

**(e) Token consumption**
Check `s["scores"]` for token counts if available. Flag extreme values (e.g. >10k tokens/sample for simple choice tasks).

### 4. Sample extraction

Pick **3–5 representative failures** covering different patterns (not all the same kind). For each, show:

```markdown
### Sample `<id>` — <short pattern label>
- **Input** (trimmed to 400 chars): `<text>`
- **Output** (trimmed to 400 chars): `<completion or "[EMPTY]">`
- **Parsed answer**: `<parsed_answer or "UNPARSED">`
- **Why it looks wrong**: <one-line observation>
```

### 5. Narrative summary

Close with 3–6 bullets answering "**어떤 경향성을 보이면서 틀리는가**":
- Top driver of failures (wrong label preference / format violation / language mixing / knowledge gap / etc.)
- Whether the benchmark itself has oddities affecting the measurement
- Concrete config/prompt change the user could try (e.g. "increase max_tokens", "disable thinking for this benchmark", "add 한국어로 답변하세요 to system prompt")

## Output format

One markdown block, this outline:

```markdown
# <config> — <benchmark> 오답 분석

## 요약
- Samples: <total>  |  Correct: <n> (<p%>)  |  Wrong: <n> (<p%>)
- Main driver: <one sentence>

## 실패 패턴

| 시그널 | 값 | 해석 |
|---|---|---|
| Answer bias | ... | ... |
| Length quartile wrong-rate | Q1 ... / Q4 ... | ... |
| Format violation (unparsed) | <n> (<p%>) | ... |
| Language mixing | <n> (<p%>) | ... |
| Token consumption | avg <n> | ... |

## 대표 실패 샘플
### Sample <id> — <pattern>
...

### Sample <id> — <pattern>
...

## 경향성 요약
- ...
- ...

## 권장 조치
- ...
```

## Rules

- Use the **full config name** in headers — never abbreviate.
- 3 decimals for percentages; 2 for rates.
- Truncate long inputs/outputs to 400 chars with a `…` marker — don't dump full text.
- **Always save** the final analysis report to `analyses/results/<config>__<benchmark>-errors.md`.
- Raw sample data goes to `analyses/data/<config>__<benchmark>.json`.
- Don't invent patterns. If no clear signal appears (wrong-rate is uniform), say "뚜렷한 편향 없음 — 벤치 자체의 난이도 영향으로 보임".

## Example invocations

- `/horangi-fails NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 kmmlu_pro, horangi/horangi4/jn7urcxl`
- "NVIDIA-Nemotron-3-Super-120B-A12B-FP8 의 ko_arc_agi 틀린 경향 분석해줘, horangi/horangi4/kagx8jup"
- Often called as a follow-up when `horangi-analyze` flags a weak (model, benchmark) pair.

## Data source

**Weave traces via `analyses/retrieve_eval.py`**. Use `find_eval_call(entity_project, run_id, benchmark)` → `get_eval_samples(entity_project, call_id)`. Cached as JSON in `analyses/data/` after first fetch.
