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
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import BaseDataset
from . import register_dataset


DEFAULT_REPO_ID = "openai/mrcr"
DEFAULT_FILENAME = "default.parquet"


TOKEN_BUCKETS: Dict[str, Tuple[int, int]] = {
    "4k": (4096, 8192),
    "8k": (8192, 16384),
    "16k": (16384, 32768),
    "32k": (32768, 65536),
    "128k": (65536, 131072),  # OpenAI MRCR naming convention
    "256k": (131072, 262144),
    "512k": (262144, 524288),
    "1m": (524288, 1048576),
}


try:  # Optional dependency for precise token counting
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore


@register_dataset("mrcr")
class MRCRDataset(BaseDataset):
    """MRCR 벤치마크용 데이터셋 로더."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_REPO_ID,
        split: str = "train",
        filename: str = DEFAULT_FILENAME,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # MRCR은 subset 개념이 없으므로 BaseDataset.subset은 사용하지 않는다.
        self.dev_mode = kwargs.pop("dev", False)
        super().__init__(dataset_name, split=split, **kwargs)
        self.filename = filename
        self.task = task.lower() if task else None
        if self.task and self.task not in TOKEN_BUCKETS:
            valid = ", ".join(sorted(TOKEN_BUCKETS))
            raise ValueError(
                f"Unsupported task '{task}'. Choose one of: {valid}"
            )
        self._encoder = None

    def _get_token_bounds(self) -> Optional[Tuple[int, int]]:
        if not self.task:
            return None
        return TOKEN_BUCKETS[self.task]

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        if tiktoken is None:
            raise ImportError(
                "task filtering requires the 'tiktoken' package."
                " Install via 'pip install tiktoken'."
            )
        self._encoder = tiktoken.get_encoding("o200k_base")

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        self._ensure_encoder()
        assert self._encoder is not None
        count = 0
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            count += len(self._encoder.encode(content))
        return count

    def load(self) -> List[Dict[str, Any]]:
        """HF Hub에서 MRCR Parquet 파일을 불러와 파이프라인 표준 포맷으로 변환."""

        dataset = load_dataset(
            path=self.dataset_name,
            data_files={self.split: self.filename},
            split=self.split,
            **self.kwargs,
        )

        samples: List[Dict[str, Any]] = []
        token_bounds = self._get_token_bounds()

        for row in dataset:
            prompt_raw = row.get("prompt")
            answer = row.get("answer", "")

            if prompt_raw is None:
                raise ValueError("MRCR row is missing 'prompt' field")

            try:
                messages = json.loads(prompt_raw)
            except json.JSONDecodeError as exc:
                raise ValueError("Failed to parse MRCR prompt JSON") from exc

            token_count = None
            if token_bounds:
                token_count = self._count_tokens(messages)
                lower, upper = token_bounds
                if not (lower <= token_count <= upper):
                    continue

            metadata = {
                "random_string_to_prepend": row.get("random_string_to_prepend", ""),
                "n_needles": row.get("n_needles"),
                "desired_msg_index": row.get("desired_msg_index"),
                "total_messages": row.get("total_messages"),
                "n_chars": row.get("n_chars"),
                "filename": self.filename,
            }
            if token_count is not None:
                metadata["token_count"] = token_count
                metadata["token_bounds"] = token_bounds

            # 예제 스크립트와 동일하게 messages 자체를 모델에 사용하고,
            # input에는 JSON 문자열 형태로 보관하여 로깅 및 few-shot 로직과 호환되도록 한다.
            sample = {
                "input": json.dumps(messages, ensure_ascii=False),
                "reference": answer,
                "metadata": metadata,
                "messages": messages,
            }

            samples.append(sample)

            if self.dev_mode and len(samples) >= 2:
                break

        return samples

    def get_raw_samples(self) -> Any:
        """Raw HuggingFace Dataset 객체 반환."""

        return load_dataset(
            path=self.dataset_name,
            data_files={self.split: self.filename},
            split=self.split,
            **self.kwargs,
        )

    def info(self) -> Dict[str, Any]:
        """MRCR 데이터셋 메타데이터 반환."""

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "filename": self.filename,
            "description": (
                "MRCR (Multi-round Co-reference Resolution) long-context benchmark"
            ),
            "license": "MIT",
            "evaluation_only": ["sequence_match"],
        }


