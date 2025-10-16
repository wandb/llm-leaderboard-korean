import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict, Any, Optional

class KodarkbenchDataset:
    def __init__(
        self,
        csv_path: str = "hf://datasets/Rice-Bobb/KoDarkBench/KoDarkBench Data (1).csv",
        model_name: str = "gpt2",
        batch_size: int = 4,
        base_prompt_template: Optional[str] = None,
        subset_name: str = "generation",
    ):
        self.csv_path = csv_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.base_prompt_template = base_prompt_template
        self.subset_name = subset_name

    def make_prompt(self, instruction: str) -> str:
        if self.base_prompt_template:
            prompt = self.base_prompt_template.format(instruction=instruction)
            print(f"[DEBUG] make_prompt with template: {prompt}")
            return prompt
        print(f"[DEBUG] make_prompt no template: {instruction}")
        return instruction

    def load(self) -> List[Dict[str, Any]]:
        print("[DEBUG] load() 시작")
        try:
            df = pd.read_csv(self.csv_path)
            print(f"[DEBUG] CSV 로딩 성공, shape: {df.shape}")
        except Exception as e:
            print(f"[ERROR] CSV 로딩 실패: {e}")
            raise

        if "instruction" in df.columns:
            print("[DEBUG] 'instruction' 컬럼 발견. 프롬프트 생성 중...")
            df["prompt"] = df["instruction"].apply(self.make_prompt)
        elif "ko-input" in df.columns:
            print("[DEBUG] 'ko-input' 컬럼 발견. 프롬프트 생성 중...")
            df["prompt"] = df["ko-input"].apply(self.make_prompt)
        else:
            err_msg = f"[ERROR] 필수 컬럼 없음: {list(df.columns)}"
            print(err_msg)
            raise ValueError("CSV must have 'instruction' or 'ko-input' column.")

        try:
            print("[DEBUG] 모델과 토크나이저 로드 시작")
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("[DEBUG] 모델과 토크나이저 로드 완료")
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            raise

        responses = []
        prompts = df["prompt"].tolist()
        print(f"[DEBUG] 총 프롬프트 수: {len(prompts)}")

        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Inference"):
            print(f"[DEBUG] 배치 {i} 시작")
            batch_prompts = prompts[i : i + self.batch_size]
            batch_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            batch_inputs = {k: v for k, v in batch_inputs.items()}
            try:
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0,
                )
                print(f"[DEBUG] 배치 {i} 추론 완료, outputs 수: {len(outputs)}")
            except Exception as e:
                print(f"[ERROR] 배치 {i} 추론 실패: {e}")
                raise

            for output in outputs:
                response_str = tokenizer.decode(output, skip_special_tokens=True)
                print(f"[DEBUG] 응답 생성: {response_str[:60]}...")
                responses.append(response_str)

        print("[DEBUG] 모든 배치 추론 완료, 응답 수:", len(responses))

        result_list = []
        for idx, row in df.iterrows():
            item = {
                "input": row["prompt"],
                "reference": "",
                "subject": row.get("subject", ""),
                "ability": row.get("ability", ""),
                "_subset_name": self.subset_name,
                "response": responses[idx] if idx < len(responses) else ""
            }
            result_list.append(item)

        print("[DEBUG] 결과 리스트 생성 완료")
        return result_list

    def info(self) -> Dict[str, Any]:
        info = {
            "csv_path": self.csv_path,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "base_prompt_template": self.base_prompt_template,
            "subset_name": self.subset_name
        }
        print(f"[INFO] 현재 설정: {info}")
        return info
