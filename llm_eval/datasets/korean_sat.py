import os
import pandas as pd
import wandb
import logging
from typing import List
from llm_eval.datasets.base import BaseDataset
from llm_eval.datasets import register_dataset
from llm_eval.utils.logging import get_logger
from dotenv import load_dotenv

load_dotenv()

WANDB_PROJECT_NAME = os.getenv("WANDB_KOREAN_SAT_PROJECT")
logger = get_logger(name="korean_sat", level=logging.INFO)

#TODO: singleton 적용, Grading info로 Calculation 하기

@register_dataset("korean_sat")
class KoreanSATDataset(BaseDataset):
    def __init__(self, subset='Korean', split="test", **kwargs):
        super().__init__(dataset_name='korean_sat', subset=subset, split=split, **kwargs)

    def load(self):
        if not WANDB_PROJECT_NAME:
            raise ValueError("WANDB_KOREAN_SAT_PROJECT environment variable is not set.")
        run = wandb.init()
        artifact = run.use_artifact(f"{WANDB_PROJECT_NAME}/korean_sat:v0", type='dataset')
        artifact_dir = artifact.download()

        df = pd.read_parquet(os.path.join(artifact_dir, "2015_2025_KoSAT.parquet"))
        samples = df.apply(self.preprocess_row, axis=1).tolist()

        # 서브셋 필터링
        if self.subset:
            if isinstance(self.subset, str):
                samples = [s for s in samples if s["_subset_name"] == self.subset]
            elif isinstance(self.subset, list):
                samples = [s for s in samples if s["_subset_name"] in self.subset]

        return samples

    def info(self):
        return {
            "description": (
                "Korean SAT dataset from 2015-2025. "
                "You can check the SAT exam year, problem number using metadata['query_id'] and metadata['question_num']."
                "metadata['score'] represents the score for each problem."
                "metadata['corpus_id'] is the identifier of the paragraph in the Korean SAT problem."
            ),
            "evaluation_only": None,
            "citation": "Korean SAT dataset",
        }

    def preprocess_row(self, row):
        prompt = self.generate_prompt_template(row['query'], row['corpus'])

        query_id = row['qid']
        corpus_id = row['corpus_id']
        question_num = row['question_index']

        # Convert answers and scores to strings
        answer = str(row['generation_gt'].split("(")[0]).strip()
        score = str(row['generation_gt'].split("(")[1].split(")")[0]).strip()

        # Validate the values
        assert answer in ["1", "2", "3", "4", "5"], "The choice_gt must be in ['1', '2', '3', '4', '5']."
        assert score in ["2", "3"], "The right score must be '2' or '3'."

        sample = {
            "input": prompt,
            "reference": answer,
            "_subset_name": 'Korean',
            "metadata": {
                "score": score,
                "query_id": query_id,
                "question_num": question_num,
                "corpus_id": corpus_id
            },
            "judge_type": "korean_sat_eval"  # Set the judge type for Korean SAT evaluation
        }

        return sample

    def get_raw_samples(self):
        run = wandb.init()
        artifact = run.use_artifact(f"{WANDB_PROJECT_NAME}/korean_sat:v0", type='dataset')
        artifact_dir = artifact.download()
        df = pd.read_parquet(os.path.join(artifact_dir, "2015_2025_KoSAT.parquet"))
        return df.to_dict('records')

    def generate_prompt_template(self, query: str, corpus: str):
        return f"""
            국어 시험 문제를 푸는 대한민국의 똑똑한 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.
            지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.

            지문 :
            {corpus}

            {query}

            문제를 풀이할 때, 반드시 지문을 참고하세요. 문제는 무조건 1개의 정답만 있습니다. 문제를 풀이할 때 모든 선택지들을 검토하세요.
            먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.
            그 다음, 문제를 해결하기 위해 그 계획에 따라 단계별로 실행하세요.

            다음의 형식을 따라 답변하세요.
            1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
            2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
            3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
            4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
            5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"
            최종 정답: (최종 정답)

            정답 :
        """

@register_dataset("korean_sat_grading_info")
class KoreanSATGradingInfoDataset(BaseDataset):
    def __init__(self, subset:List[str]=None, split="test", **kwargs):
        super().__init__(dataset_name='korean_sat_grading_info', subset=subset, split=split, **kwargs)

    def load(self):
        if not WANDB_PROJECT_NAME:
            raise ValueError("WANDB_KOREAN_SAT_PROJECT environment variable is not set.")
        run = wandb.init()
        artifact = run.use_artifact(f"{WANDB_PROJECT_NAME}/korean_sat_grading_info:v0", type='dataset')
        artifact_dir = artifact.download()

        import pandas as pd
        from pathlib import Path
        folder_path = Path(artifact_dir)
        grade_info = {
            file.stem: pd.read_csv(file, encoding='utf-8-sig')  # 한글 인코딩 처리
            for file in folder_path.glob("*.csv")
        }

        if self.subset:
            grade_info = {year: grade_info[year] for year in self.subset if year in grade_info}
        return grade_info

    def info(self):
        return {
            "description": (
                "Korean SAT dataset grade information from 2015-2025 for comparing between human and llm."
                "This dataset includes raw score(원점수), scaled score(표준점수), percentile(백분위), and grade(등급) during 2014-2025."
            ),
            "evaluation_only": None,
            "citation": "Korean SAT grading info dataset",
        }

