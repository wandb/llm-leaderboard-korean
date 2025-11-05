"""MRCR Dataset Loader

이 모듈은 OpenAI MRCR 데이터셋(https://huggingface.co/datasets/openai/mrcr)을
HRET 파이프라인에서 사용할 수 있도록 변환합니다. MRCR은 긴 대화 컨텍스트
내에 동일한 요청(needle)을 여러 번 삽입하여 특정 순번의 응답을 찾는 모델의
장기 문맥 추적 능력을 평가하는 벤치마크입니다.

샘플 포맷:
    {
        "input": List[Dict[str, str]]  # OpenAI ChatCompletion 호환 메시지 형식
        "reference": str               # 모델이 반환해야 하는 정답 문자열
        "metadata": {
            "random_string_to_prepend": str,
            "n_needles": int,
            "desired_msg_index": int,
            "total_messages": int,
            "n_chars": int,
            "bin_index": Optional[int],
            "filename": str
        }
    }

`input`은 JSON 문자열로 저장된 multi-turn 대화를 복원한 리스트입니다. 이는
OpenAI/LiteLLM 백엔드에서 그대로 사용할 수 있습니다. 다른 모델 백엔드에서도
system/user/assistant 역할 정보를 활용한 프롬프트 구성에 참고할 수 있습니다.

사용자 파라미터:
    filename: str (Optional)
        기본값은 MRCR의 대표 파일 "default.parquet"를 사용합니다. 특정 needle
        수에 해당하는 파일(예: "2needle.parquet")을 지정할 수 있습니다.
        파일은 HF Hub에서 다운로드됩니다.
    repo_id: str (Optional)
        기본값 "openai/mrcr". 포크나 미러를 사용할 경우 덮어쓸 수 있습니다.
    split: str
        Hugging Face Datasets가 제공하는 split. (현재 MRCR은 train만 제공)

참고: MIT 라이선스로 배포되므로 배포 시 라이선스 고지를 유지해야 합니다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseDataset
from . import register_dataset


@register_dataset("mrcr")
class MRCRDataset(BaseDataset):
    """MRCR 벤치마크용 데이터셋 로더."""

    def __init__(
        self,
        dataset_name: str = "mrcr_2_needles",
        subset: str = "128k",
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)
        # subset은 4k/8k/16k... 같은 레이블 용도이며, 필터링에는 사용하지 않음

    def _normalize_split(self, split: str) -> str:
        # MRCR는 train 키를 사용합니다(아티팩트 구조 기준)
        return "train"

    def _download_and_load(self) -> Dict[str, Any]:
        from llm_eval.wandb_singleton import WandbConfigSingleton
        artifact_dir = WandbConfigSingleton.download_artifact(self.dataset_name)
        file_path = os.path.join(artifact_dir, f"{self.dataset_name}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid {self.dataset_name}.json format: expected an object keyed by splits")
        return data

    # 토큰 카운팅/필터링 관련 코드는 더 이상 사용하지 않습니다.

    def load(self) -> List[Dict[str, Any]]:
        """W&B artifact의 mrcr.json을 불러와 파이프라인 표준 포맷으로 변환."""

        raw = self._download_and_load()
        split_key = self._normalize_split(self.split)
        split_obj = raw.get(split_key, [])
        # 지원: {split: {subset: [...]}} 또는 {split: [...]}
        if isinstance(split_obj, dict):
            if isinstance(self.subset, (list, tuple)):
                merged: List[Dict[str, Any]] = []
                for subset_name in self.subset:
                    items = split_obj.get(subset_name, [])
                    if isinstance(items, list):
                        merged.extend(items)
                dataset = merged
            else:
                dataset = split_obj.get(self.subset, [])
        else:
            dataset = split_obj if isinstance(split_obj, list) else []

        samples: List[Dict[str, Any]] = []

        for row in dataset:
            prompt_raw = row.get("prompt")
            answer = row.get("answer", "")

            if prompt_raw is None:
                raise ValueError("MRCR row is missing 'prompt' field")

            try:
                messages = json.loads(prompt_raw)
            except json.JSONDecodeError as exc:
                raise ValueError("Failed to parse MRCR prompt JSON") from exc
            metadata = {
                "random_string_to_prepend": row.get("random_string_to_prepend", ""),
                "n_needles": row.get("n_needles"),
                "desired_msg_index": row.get("desired_msg_index"),
                "total_messages": row.get("total_messages"),
                "n_chars": row.get("n_chars"),
            }

            # 예제 스크립트와 동일하게 messages 자체를 모델에 사용하고,
            # input에는 JSON 문자열 형태로 보관하여 로깅 및 few-shot 로직과 호환되도록 한다.
            sample = {
                "input": json.dumps(messages, ensure_ascii=False),
                "reference": answer,
                "metadata": metadata,
                "messages": messages,
            }

            samples.append(sample)
            if self.dev and len(samples) >= self.limit:
                break
            if len(samples) >= self.num_samples:
                break
        return samples

    def get_raw_samples(self) -> Any:
        return self._download_and_load()

    def info(self) -> Dict[str, Any]:
        """MRCR 데이터셋 메타데이터 반환."""

        return {
            "dataset_name": self.dataset_name,
            "split": self._normalize_split(self.split),
            "subset": self.subset,
            "description": "MRCR, long-context benchmark loaded from W&B artifact (artifact name selects needles)",
        }


@register_dataset("mrcr_2_needles")
class MRCR2NeedlesDataset(MRCRDataset):
    def __init__(
        self,
        dataset_name: str = "mrcr_2_needles",
        subset: str = "128k",
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset_name=dataset_name, subset=subset, split=split, **kwargs)


@register_dataset("mrcr_4_needles")
class MRCR4NeedlesDataset(MRCRDataset):
    def __init__(
        self,
        dataset_name: str = "mrcr_4_needles",
        subset: str = "128k",
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset_name=dataset_name, subset=subset, split=split, **kwargs)


@register_dataset("mrcr_8_needles")
class MRCR8NeedlesDataset(MRCRDataset):
    def __init__(
        self,
        dataset_name: str = "mrcr_8_needles",
        subset: str = "128k",
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset_name=dataset_name, subset=subset, split=split, **kwargs)