import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationOutput
from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger

logger = get_logger(name="runner", level=logging.INFO)

@register_model("huggingface")
class HuggingFaceModel(BaseModel):
    """
    HuggingFace Transformers 기반 모델 백엔드. 현재 batch 처리는 미지원
    
    특징:
      - return_logits=True 시, generate(..., return_dict_in_generate=True, output_scores=True)로
        토큰별 logits -> log_prob를 간단히 계산 후, item["logits"]에 저장
        (ex: "logits": {"sum_log_prob": float, "token_log_probs": [...], "tokens": [...]} )
      - cot_parser callable을 통해 chain_of_thought와 최종 answer를 분리 가능.
        (예: regex, "Final Answer:" 구분자 등)
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
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = None,
        **kwargs
    ):
        """
        Args:
            model_name_or_path (str): HF Hub 모델 ID 또는 로컬 경로
            device (str): 'cpu', 'cuda', 'cuda:0' 등
            max_new_tokens (int): 한 번의 generate에서 생성할 최대 토큰
            cot_trigger (str|None): CoT 유도 문구. None이면 CoT 트리거 없음
            temperature (float): 샘플링 온도
            top_p (float): nucleus 샘플링
            do_sample (bool): True면 샘플링 모드, False면 greedy
            cot_parser (callable|None): (generated_text)->(chain_of_thought, final_answer)
                                        형태로 파싱하는 함수
            kwargs: 기타 필요한 파라미터(logging 등)
        """
        super().__init__(**kwargs)
        logger.info(f"[HuggingFaceModel] Loading tokenizer/model from {model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.eval()

        self.device = device
        if self.device != "cpu":
            self.model.to(self.device)

        self.max_new_tokens = max_new_tokens
        self.cot_trigger = cot_trigger
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.cot_parser = cot_parser  # 외부에서 주입된 파싱 함수

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        cot: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        (단일 샘플씩) 모델에 입력 -> 텍스트 생성 -> 결과 업데이트.
        
        Args:
            inputs: [{"input": str, "reference": str, ...}, ...]
            return_logits (bool): True면 logits(or log_probs) 계산하여 item["logits"]에 저장
            cot (bool): True면 self.cot_trigger를 prompt 끝에 추가 + (cot_parser가 있으면) 분리 파싱

        Returns:
            동일 리스트(또는 복사본)에
            [
                {
                  "input": ...,
                  "reference": ...,
                  "prediction": <생성된 최종 답>,
                  "chain_of_thought": <cot_parser로 분리된 Reasoning> (옵션),
                  "logits": { "sum_log_prob": float, "token_log_probs": [...], "tokens": [...] } (옵션),
                  ...
                },
                ...
            ]
        """
        out_list = []
        for item in inputs:
            prompt = item["input"]
            if cot and self.cot_trigger:
                prompt = f"{prompt}\n{self.cot_trigger}\n"

            # Tokenize
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
            )
            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
            }

            if return_logits:
                # output_scores=True + return_dict_in_generate=True로 logits 계산
                gen_kwargs.update({
                    "return_dict_in_generate": True,
                    "output_scores": True,
                })

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    **gen_kwargs,
                )

            # outputs가 GenerationOutput 또는 텐서형태로 반환됨
            if return_logits and isinstance(outputs, GenerationOutput):
                # HF 4.26+ 에서: outputs.sequences, outputs.scores 등
                #   sequences: (batch_size, total_sequence_len)
                #   scores: list of length=#new_tokens, each shape: (batch_size, vocab_size)
                sequences = outputs.sequences
                scores = outputs.scores  # List[Tensor], length = #generated_tokens
            else:
                # 이전 버전 호환
                if isinstance(outputs, GenerationOutput):
                    sequences = outputs.sequences
                    scores = None
                else:
                    # 구버전 호환: returns just a Tensor
                    sequences = outputs
                    scores = None

            # input 길이
            input_len = encoded["input_ids"].shape[1]
            gen_ids = sequences[0][input_len:]  # 생성 부분
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            # CoT 파싱
            final_answer = generated_text
            chain_of_thought = None

            if cot and self.cot_parser is not None:
                # 사용자가 정의한 파싱 함수: text -> (chain_of_thought, final_answer)
                chain_of_thought, final_answer = self.cot_parser(generated_text)

            item["prediction"] = final_answer
            if chain_of_thought is not None:
                item["chain_of_thought"] = chain_of_thought

            # logits 계산
            if return_logits and scores is not None:
                # scores: length=#generated_tokens, each shape=(batch_size, vocab_size)
                # 여기서는 batch_size=1 만 일단 고려 (아직 batch 처리 미지원원)
                log_probs_per_token = []
                sum_log_prob = 0.0
                token_ids = gen_ids.tolist()  # 생성된 token id list

                for i, step_scores in enumerate(scores):
                    # step_scores shape: (1, vocab_size)
                    token_id = token_ids[i]
                    # log softmax
                    step_log_probs = F.log_softmax(step_scores[0], dim=-1)  # (vocab_size,)
                    token_log_prob = step_log_probs[token_id].item()
                    sum_log_prob += token_log_prob
                    log_probs_per_token.append(token_log_prob)

                item["logits"] = {
                    "sum_log_prob": sum_log_prob,
                    "token_log_probs": log_probs_per_token,
                    "tokens": self.tokenizer.convert_ids_to_tokens(token_ids),
                }

            out_list.append(item)

        return out_list
