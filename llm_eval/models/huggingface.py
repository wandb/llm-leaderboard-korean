import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationOutput
from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger
from llm_eval.utils.prompt_template import extract_final_answer

logger = get_logger(name="huggingface", level=logging.INFO)

@register_model("huggingface")
class HuggingFaceModel(BaseModel):
    """
    HuggingFace Transformers 기반 모델 백엔드.
    
    특징:
      - return_logits=True 시, generate(..., return_dict_in_generate=True, output_scores=True)를 통해
        토큰별 logits -> log_prob를 계산 후, item["logits"]에 저장
      - cot_parser callable을 통해 chain_of_thought와 최종 answer를 분리 가능.
    
    Args:
        model_name_or_path (str): HF Hub 모델 ID 또는 로컬 경로
        device (str): 'cpu', 'cuda', 'cuda:0' 등
        max_new_tokens (int): 한 번의 generate에서 생성할 최대 토큰 수
        cot_trigger (str|None): CoT 유도 문구. None이면 CoT 트리거 없음
        temperature (float): 샘플링 온도
        top_p (float): nucleus 샘플링 확률
        do_sample (bool): True면 샘플링 모드, False면 탐욕적(greedy) 생성
        cot_parser (callable|None): (generated_text)->(chain_of_thought, final_answer) 형태로 파싱하는 함수
        **kwargs: 기타 필요한 파라미터
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        max_new_tokens: int = 128,
        cot_trigger: Optional[str] = "Let's think step by step.",
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = extract_final_answer,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f"[HuggingFaceModel] Loading tokenizer/model from {model_name_or_path}")

        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.eval()

        # 디바이스 설정
        self.device = device
        if self.device != "cpu":
            self.model.to(self.device)
            logger.info(f"[HuggingFaceModel] Model moved to {self.device}")

        # 추론 하이퍼파라미터 설정
        self.max_new_tokens = max_new_tokens
        self.cot_trigger = cot_trigger
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.cot_parser = cot_parser

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        cot: bool = False,
        batch_size: Optional[Union[int, str]] = "auto"
    ) -> List[Dict[str, Any]]:
        """
        배치 단위로 모델에 입력 -> 텍스트 생성 -> 결과 업데이트.
        
        Args:
            inputs (List[Dict[str, Any]]): [{"input": str, "reference": str, ...}, ...]
            return_logits (bool): True면 logits(or log_probs)을 계산하여 item["logits"]에 저장
            cot (bool): True면 self.cot_trigger를 prompt 끝에 추가 + (cot_parser가 있으면) 분리 파싱
            batch_size (int | str): 사용할 배치 크기. 정수 지정하거나 "auto"로 자동 탐색.

        Returns:
            List[Dict[str, Any]]: 입력 리스트와 동일한 구조의 리스트로 각 항목에 대해 다음 필드를 추가:
                - "prediction": 생성된 최종 답
                - "chain_of_thought": (옵션) CoT 파싱 결과
                - "logits": { "sum_log_prob": float, "token_log_probs": [...], "tokens": [...] } (옵션)
        """
        # 배치 사이즈 설정
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            auto_mode = True
            current_bs = 128  # "auto" 모드의 초기 배치 사이즈
            logger.info(f"[HuggingFaceModel] Batch size set to 'auto'. Starting with batch size {current_bs}.")
        else:
            auto_mode = False
            current_bs = batch_size if batch_size is not None else len(inputs)
            logger.info(f"[HuggingFaceModel] Batch size set to {current_bs}.")

        while True:
            try:
                results = []
                # 입력 데이터를 current_bs 크기로 분할하여 처리
                for start in range(0, len(inputs), current_bs):
                    batch_items = inputs[start:start + current_bs]
                    
                    # 프롬프트 구성
                    prompts = [
                        f"{item['input']}\n{self.cot_trigger}\n" if cot and self.cot_trigger else item["input"]
                        for item in batch_items
                    ]

                    # 배치 단위 토큰화 (패딩 및 트렁케이션 포함)
                    encoded = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )

                    # 각 샘플의 실제 입력 길이 계산
                    input_lens = encoded["attention_mask"].sum(dim=1).tolist()

                    # 디바이스로 이동
                    if self.device != "cpu":
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}

                    # 생성 인자 설정
                    gen_kwargs = {
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "do_sample": self.do_sample,
                    }
                    if return_logits:
                        gen_kwargs.update({
                            "return_dict_in_generate": True,
                            "output_scores": True,
                        })

                    # 배치 단위 텍스트 생성
                    with torch.no_grad():
                        outputs = self.model.generate(**encoded, **gen_kwargs)

                    # GenerationOutput 처리
                    if return_logits and isinstance(outputs, GenerationOutput):
                        sequences = outputs.sequences  # shape: (batch_size, sequence_length)
                        scores = outputs.scores        # list of length=#generated_tokens, each shape: (batch_size, vocab_size)
                    else:
                        if isinstance(outputs, GenerationOutput):
                            sequences = outputs.sequences
                            scores = None
                        else:
                            sequences = outputs
                            scores = None

                    batch_size_actual = sequences.shape[0]

                    # 배치 내 각 샘플에 대해 결과 처리
                    for i in range(batch_size_actual):
                        item = batch_items[i]
                        input_len = input_lens[i]
                        gen_ids = sequences[i][input_len:]  # 생성된 부분 추출
                        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                        # CoT 파싱
                        final_answer = generated_text
                        chain_of_thought = None
                        if cot and self.cot_parser is not None:
                            chain_of_thought, final_answer = self.cot_parser(generated_text)

                        item["prediction"] = final_answer
                        if chain_of_thought is not None:
                            item["chain_of_thought"] = chain_of_thought

                        # logits 계산
                        if return_logits and scores is not None:
                            log_probs_per_token = []
                            sum_log_prob = 0.0
                            token_ids = gen_ids.tolist()

                            # 각 시간 단계별로 해당 샘플에 대한 log_prob 계산
                            for t, step_scores in enumerate(scores):
                                if t >= len(token_ids):
                                    break
                                step_score = step_scores[i]  # 해당 샘플의 점수
                                token_id = token_ids[t]
                                step_log_probs = F.log_softmax(step_score, dim=-1)
                                token_log_prob = step_log_probs[token_id].item()
                                sum_log_prob += token_log_prob
                                log_probs_per_token.append(token_log_prob)

                            item["logits"] = {
                                "sum_log_prob": sum_log_prob,
                                "token_log_probs": log_probs_per_token,
                                "tokens": self.tokenizer.convert_ids_to_tokens(token_ids),
                            }

                        results.append(item)

                return results
            except RuntimeError as e:
                # Decrease batch size upon OOM
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if auto_mode:
                        if current_bs > 1:
                            current_bs = max(1, current_bs // 2)
                            logger.warning(f"OOM detected: reducing batch size to {current_bs}.")
                        else:
                            logger.error("Batch size is already 1, but OOM still occurs.")
                            raise RuntimeError("Out of memory even with batch size = 1.") from e
                    else:
                        logger.error("Out of memory for the specified batch size.")
                        raise RuntimeError("Out of memory for the specified batch size.") from e
                else:
                    logger.error("A RuntimeError occurred:", exc_info=True)
                    raise
