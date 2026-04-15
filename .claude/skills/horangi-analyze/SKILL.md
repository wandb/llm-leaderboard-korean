---
name: horangi-analyze
description: Analyze one or more Horangi model configs against (1) HuggingFace model card claims, (2) the W&B leaderboard rankings, and (3) category-level peer comparison. Invoke when the user asks to analyze/compare model performance for a config (e.g. "analyze <config>", "compare X and Y", "이 모델 성능 분석해줘").
---

# horangi-analyze

Produce a side-by-side analysis of 1..N Horangi model configs. Output goes to chat as markdown — **never create a file** unless the user explicitly asks.

## Inputs

- One or more config names (YAML filename in `configs/models/` without `.yaml`).
- W&B `entity/project` for the leaderboard. **Always ask** the user at the start — do not default to any specific project. `.env` values (`WANDB_ENTITY`, `WANDB_PROJECT`) can be offered as a suggestion, but confirm before using.
- If any input is missing or ambiguous, ask — don't guess.

## Workflow

### 1. Resolve configs

For each input config:
- Read `configs/models/<name>.yaml`
- Extract: `wandb.run_name`, `model.name` (HF ID or hosted model ID), `metadata.model_size`, `metadata.active_params`, `metadata.context_window`, `metadata.release_date`.
- If `model.name` looks like a HuggingFace repo (`org/model-id`), that's the HF URL path. For hosted-only models (W&B Inference, OpenRouter slugs), the **HF upstream** may be mentioned in YAML comments or need a quick web search.

### 2. Fetch HF model card claims (WebFetch)

For each HF repo:
```
https://huggingface.co/<org>/<model-id>
```

Extract every numeric benchmark the card reports. Organize by category (reasoning / coding / knowledge / agentic / instruction-following / safety / long-context / multilingual). Capture variants (e.g. "AIME25 no tools" vs "AIME25 with tools", "SWE-Bench Verified OpenHands" vs "Codex"). Note the reported sampling settings if present.

Do **not** fabricate numbers — only what's on the page.

### 3. Query W&B leaderboard (wandb API)

Use this snippet to pull leaderboard-tagged runs:

```python
import os
os.environ.setdefault('WANDB_SILENT', 'true')
from dotenv import load_dotenv
load_dotenv('.env', override=True)
import wandb, pandas as pd

api = wandb.Api()
# <entity>/<project> supplied by the user at invocation time
runs = list(api.runs(f'{entity}/{project}', filters={'tags': {'$in': ['leaderboard']}}))

rows = []
for r in runs:
    try:
        for art in r.logged_artifacts():
            if 'leaderboard_table' in art.name:
                t = art.get('leaderboard_table')
                if t:
                    df = t.get_dataframe()
                    df['run_name'] = r.name
                    rows.append(df)
                    break
    except Exception:
        pass

big = pd.concat(rows, ignore_index=True).drop_duplicates(subset='run_name', keep='last')
big = big.sort_values('FINAL_SCORE', ascending=False).reset_index(drop=True)
big['rank'] = big.index + 1
```

`entity` and `project` must come from the user's invocation — the skill never assumes a specific W&B path.

`leaderboard_table` columns include: `FINAL_SCORE`, `범용언어성능(GLP)_AVG`, `가치정렬성능(ALT)_AVG`, and 16 category columns (`GLP_*`, `ALT_*`).

For each target model, compute:
- Overall rank (out of total leaderboard runs)
- Per-category rank
- Peer comparison (filter similar open-source models by size/family)

Also fetch `benchmark_detail_table` for per-benchmark raw scores if a deeper HF-vs-measured comparison is needed:

```python
for art in r.logged_artifacts():
    if 'benchmark_detail_table' in art.name:
        detail_df = art.get('benchmark_detail_table').get_dataframe()
        break
```

### 4. Match benchmarks for HF-vs-measured comparison

Common overlaps (not exhaustive):

| HF benchmark | Horangi benchmark | Notes |
|---|---|---|
| AIME25 (no tools) | `ko_aime2025` | Korean translation — typically -5~10% |
| HLE | `ko_hle` | Korean subset |
| MMLU-Pro | `kmmlu_pro` | **Different benchmark** — KMMLU-Pro = Korean domain knowledge, not translated MMLU-Pro. Expect -15~25% gap. |
| SWE-Bench Verified | `swebench_verified_official_80` | 80-sample subset, variance higher |
| BFCL (v4) | `bfcl` | Horangi uses text-based variant |
| IFBench (prompt) | `ifeval_ko` | **Different** — IFEval Korean ≠ IFBench |
| HumanEval+ | `humaneval_100` | 100-sample subset |
| BigCodeBench | `bigcodebench_100` | 100-sample subset |

Call these out explicitly when deltas look surprising — often it's a benchmark variant difference, not model behavior.

### 5. Output format

A single markdown block. Structure scales with N:

**For N=1:**
- Section 1: "HF claim vs horangi 측정"  (comparison table of comparable benchmarks)
- Section 2: "리더보드 순위" (overall rank + category rank table)
- Section 3: "동급 피어 비교" (similar-size open-source models)
- Section 4: "결론" (3-5 bullets)

**For N≥2:**
- Section 1: "모델 비교" (side-by-side table, all Horangi benchmarks)
- Section 2: "HF claim vs 측정" (per-model comparison table)
- Section 3: "리더보드 순위" (all models' ranks side by side)
- Section 4: "카테고리별 우위" (which model wins which category)
- Section 5: "결론"

Rules:
- Use **full model name** (e.g. `NVIDIA-Nemotron-3-Super-120B-A12B-FP8`) — never abbreviate.
- Mark deltas ≥0.05 with **bold**, ≥0.10 with 🔥 or ⚠️ based on direction.
- When a Horangi score suggests the benchmark variant differs from HF, say so in the Notes/결론 — don't silently compare apples to oranges.
- Rank categories by how informative they are for the model; show top-5 strengths and bottom-3 weaknesses per model, not every category.
- Keep numbers to 3 decimals (0.641 not 0.6413).
- Cite W&B run URLs at the end: `https://wandb.ai/<entity>/<project>/runs/<run_id>`.

## Follow-up: suggest `horangi-fails`

After producing the analysis, **inspect category rankings** for each target model. If any category meets either criterion:

- **Absolute**: rank falls in the bottom 25% of the leaderboard (e.g. rank 68+ out of 90)
- **Relative**: score is ≥ 0.15 below the model's own category median (outlier weakness)

...append a "🔎 **깊이 분석 권장**" section listing the specific (model, benchmark) pairs to drill into, and suggest:

> 이 영역이 유독 낮아 보여요. 오답 경향을 자세히 보고 싶으면 `/horangi-fails <config> <benchmark>`로 불러주세요.

Map each weak **category** back to the underlying **benchmark(s)** from `ALL_BENCHMARKS`:

| 카테고리 | 대표 벤치마크 |
|---|---|
| GLP_일반적지식 / 전문적지식 | `kmmlu`, `kmmlu_pro`, `haerae_bench_v1_*` |
| GLP_수학적추론 | `hrm8k`, `ko_aime2025` |
| GLP_논리적추론 / 추상적추론 | `ko_arc_agi`, `ko_hle` |
| GLP_코딩능력 | `humaneval_100`, `bigcodebench_100`, `swebench_verified_official_80` |
| GLP_함수호출 | `bfcl` |
| GLP_의미해석 / 구문해석 | `ko_balt_700_semantic`, `ko_balt_700_syntax`, `squad_kor_v1`, `ko_hellaswag` |
| ALT_유해성방지 / 윤리·도덕 | `korean_hate_speech`, `ko_moral`, `kobbq` |
| ALT_환각방지 | `ko_hallulens_wikiqa`, `ko_hallulens_nonexistent`, `ko_truthful_qa` |
| ALT_제어성 | `ifeval_ko`, `ko_mtbench` |

Don't propose `horangi-fails` for benchmarks that already score well, or for every weak category — pick the **1–3 most informative** drill-down candidates.

## Example invocations

- `/horangi-analyze NVIDIA-Nemotron-3-Super-120B-A12B-FP8` → skill asks for entity/project
- `/horangi-analyze NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --entity myteam --project my-bench`
- "Qwen3-30B-A3B-Instruct-2507 분석해줘" → skill asks for entity/project before proceeding

## What NOT to do

- Don't create files unprompted — chat output only. The user will say "md로 저장해줘" if they want a file, in which case use `analyses/results/<model>-analysis.md`.
- Don't compare HF English benchmark numbers directly with Korean Horangi numbers without calling out the language/variant gap.
- Don't skip the HF fetch — the "NVIDIA 주장과 일치/불일치" insight is the whole point of this skill.
- Don't assume any entity/project. Always confirm at the start of the skill invocation.
