from typing import List, Dict, Any, Optional, Union
import os
import json
from pathlib import Path

from .base import BaseDataset
from . import register_dataset


@register_dataset("mt_bench")
class MTBenchDataset(BaseDataset):
    """
    MT Bench 데이터셋 로더.

    - 기본적으로 W&B Artifact에 업로드된 `mt_bench.json` 파일을 내려받아 로드합니다.
    - 다양한 JSON 구조를 관대하게 허용합니다.
      지원 예:
        1) {"test": {"default": [ {"question": ..., "rubric": ..., "category": ...}, ... ]}}
        2) {"test": [ {...}, {...} ]}
        3) [ {...}, {...} ]

    - 표준 포맷으로 변환 시:
        input: 질문 또는 multi-turn을 하나의 프롬프트로 구성
        reference: "" (채점 기준상 정답 없음)
        rubric: 심사 기준 텍스트 (가능하면 원본 제공값 사용)
        judge_type: "rubric_and_response" (LLMJudgeEvaluator에서 점수 산출)
        _subset_name: 카테고리(있다면)
        metadata: 원본 필드 일부 보존

    - 평가 방식 제약:
        LLM-as-a-Judge가 필요하므로 `evaluation_only`에 ["llm_judge"]를 명시합니다.
    """

    def __init__(
        self,
        dataset_name: str = "mt_bench",
        subset: Optional[Union[str, List[str]]] = "default",
        split: str = "test",
        base_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        if base_prompt_template is None:
            # MT-Bench는 보통 자유서술형 답변을 평가하므로 간단한 기본 템플릿을 둡니다.
            base_prompt_template = (
                "다음 질문에 성실하고 간결하게 답변하세요.\n\n{query}"
            )
        super().__init__(
            dataset_name,
            split=split,
            subset=subset,
            base_prompt_template=base_prompt_template,
            **kwargs,
        )

    def _load_local_or_download(self) -> Dict[str, str]:
        """
        세 개의 한국어 MT Bench 아티팩트 경로를 반환합니다.
        우선 레포 내 체크인된 경로를 확인하고, 없으면 W&B에서 다운로드합니다.
        반환키: {"question", "referenceanswer", "judge_prompt"} → 각 디렉터리 경로
        """
        # 1) 레포 내 로컬 경로 우선
        root = Path(__file__).resolve().parents[2]  # 프로젝트 루트
        local_base = root / "artifacts" / "artifacts"
        candidates = {
            "question": local_base / "mtbench_ko_question:v0",
            "referenceanswer": local_base / "mtbench_ko_referenceanswer:v0",
            "judge_prompt": local_base / "mtbench_ko_prompt:v0",
        }
        paths: Dict[str, str] = {}
        if all((p.exists() for p in candidates.values())):
            for k, p in candidates.items():
                paths[k] = str(p)
            return paths

        # 2) W&B 아티팩트 다운로드
        from llm_eval.wandb_singleton import WandbConfigSingleton
        paths["question"] = WandbConfigSingleton.download_artifact("mtbench_ko_question")
        paths["referenceanswer"] = WandbConfigSingleton.download_artifact("mtbench_ko_referenceanswer")
        paths["judge_prompt"] = WandbConfigSingleton.download_artifact("mtbench_ko_prompt")
        return paths

    def _normalize_split(self, split: str) -> str:
        s = (split or "").lower()
        # MT-Bench는 보통 평가용 단일 split을 가정
        if s in ("train", "training"):
            return "train"
        if s in ("dev", "validation", "valid", "val"):
            return "dev"
        return "test"

    @staticmethod
    def _join_turns(item: Dict[str, Any]) -> str:
        """
        MT-Bench 항목이 multi-turn(예: {"turns": ["Q1", "Q2", ...]})을 가질 경우
        하나의 프롬프트 텍스트로 합칩니다. 그렇지 않으면 `question`/`query`/`prompt` 중 하나를 사용합니다.
        """
        turns = item.get("turns")
        if isinstance(turns, list) and turns:
            # 단순 결합: 줄바꿈으로 연결
            return "\n\n".join(str(t) for t in turns)

        for key in ("question", "query", "prompt", "input"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    @staticmethod
    def _extract_rubric(item: Dict[str, Any]) -> str:
        # 가능한 키 후보에서 rubric 텍스트를 찾습니다.
        for key in ("rubric", "grading_rubric", "instruction", "criteria"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
        # 기본 루브릭 (심사 모델 프롬프트가 자체적으로 가이드를 포함할 수 있으나, 명시적으로 둡니다)
        return (
            "다음 응답을 완전성, 정확성, 논리성, 표현력을 기준으로 1~10 사이의 점수로 평가하세요."
        )

    def _read_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return items

    def load(self) -> List[Dict[str, Any]]:
        # Load three Korean MT-Bench artifacts
        paths = self._load_local_or_download()

        questions_path = os.path.join(paths["question"], "question.jsonl")
        refs_path = os.path.join(paths["referenceanswer"], "gpt-4.jsonl")
        prompts_path = os.path.join(paths["judge_prompt"], "judge_ko_prompts.jsonl")

        questions = self._read_jsonl(questions_path)
        reference_items = self._read_jsonl(refs_path)
        prompt_defs = self._read_jsonl(prompts_path)

        # Map: prompt name -> template string
        name_to_template: Dict[str, str] = {}
        for p in prompt_defs:
            name = p.get("name")
            tpl = p.get("prompt_template")
            if isinstance(name, str) and isinstance(tpl, str):
                name_to_template[name] = tpl

        # Map: question_id -> reference answers (list of turns)
        qid_to_ref: Dict[int, List[str]] = {}
        for r in reference_items:
            qid = r.get("question_id")
            choices = (r.get("choices") or [])
            if isinstance(qid, int) and isinstance(choices, list) and choices:
                turns = (choices[0] or {}).get("turns") or []
                if isinstance(turns, list):
                    qid_to_ref[qid] = [str(x) for x in turns]

        # Prepare allowed subset filter (categories)
        allowed_subsets = None
        if isinstance(self.subset, str):
            if self.subset and self.subset.lower() != "default":
                allowed_subsets = {self.subset.lower()}
        elif isinstance(self.subset, (list, tuple, set)):
            allowed_subsets = {str(s).lower() for s in self.subset}

        # Build samples (single-turn focus: use turns[0])
        results: List[Dict[str, Any]] = []
        # When dev_mode is enabled, collect up to 2 samples per category (subset)
        category_counts: Dict[str, int] = {cat_key: 0 for cat_key in allowed_subsets}
        for q in questions:
            try:
                qid = q.get("question_id")
                category = q.get("category")
                turns = q.get("turns") or []
                if not isinstance(turns, list) or not turns:
                    continue

                # Determine category key for counting (normalize None)
                cat_key = str(category) if category is not None else "uncategorized"
                cat_norm = str(category).lower() if category is not None else "uncategorized"

                # Filter by allowed subsets if provided
                if allowed_subsets is not None and cat_norm not in allowed_subsets:
                    continue

                question_1 = str(turns[0])
                model_input = (
                    self.base_prompt_template.format(query=question_1)
                    if self.base_prompt_template
                    else question_1
                )

                # Pick judge template name by category
                if str(category).lower() == "math":
                    judge_name = "single-math-v1"
                else:
                    judge_name = "single-v1"
                judge_tpl = name_to_template.get(judge_name)

                # Reference (if available)
                ref_turns = qid_to_ref.get(qid, [])
                ref_answer_1 = ref_turns[0] if ref_turns else None

                sample: Dict[str, Any] = {
                    "input": model_input,
                    "reference": "",  # generation 정답 없음
                    "judge_type": "rubric_and_response",  # 파서 호환을 위해 유지
                    "_subset_name": cat_norm,
                    "question_1": question_1,
                    "metadata": {
                        "question_id": qid,
                        "category": category,
                    },
                }
                if judge_tpl:
                    sample["judge_prompt_template"] = judge_tpl
                if ref_answer_1:
                    sample["ref_answer_1"] = ref_answer_1

                # Track per-category sample count in dev_mode
                if not self.dev:
                    if category_counts[cat_key] < self.num_samples:
                        results.append(sample)
                        category_counts[cat_key] = category_counts.get(cat_key, 0) + 1
                else:
                    if category_counts[cat_key] < self.limit:
                        results.append(sample)
                        category_counts[cat_key] = category_counts.get(cat_key, 0) + 1
            except Exception:
                continue
        print(category_counts)
        return results

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def info(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": "MT Bench dataset loaded from Weights & Biases artifact.",
            # LLM-as-a-Judge 전용
            "evaluation_only": ["mt_bench_judge"],
        }
