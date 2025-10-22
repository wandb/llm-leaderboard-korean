from typing import Optional, Any, List
from types import SimpleNamespace
import pandas as pd
import wandb

try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore


class WandbConfigSingleton:
    _instance: Optional[SimpleNamespace] = None

    @classmethod
    def get_instance(cls) -> SimpleNamespace:
        if cls._instance is None:
            raise Exception("WandbConfigSingleton has not been initialized")
        return cls._instance

    @classmethod
    def initialize(cls, run, llm: Optional[Any] = None, wandb_params: Optional[Any] = None):
        if cls._instance is not None:
            raise Exception("WandbConfigSingleton has already been initialized")
        config_dict = dict(getattr(run, "config", {}) or {})
        if OmegaConf is not None:
            config = OmegaConf.create(config_dict)
        else:
            config = config_dict
        cls._instance = SimpleNamespace(run=run, config=config, blend_config=None, llm=llm, wandb_params=wandb_params, leaderboard_tables={})

    @classmethod
    def download_artifact(cls, dataset_name: str):
        api = wandb.Api()
        if "mt_bench" in dataset_name:
            artifact = api.artifact(f"wandb-korea/korean-llm-leaderboard/{dataset_name}:latest")
            artifact_path = artifact.download()
            return artifact_path
        artifact = api.artifact(f"{cls._instance.wandb_params.get('entity')}/{cls._instance.wandb_params.get('project_dataset')}/{dataset_name}:latest")
        artifact_path = artifact.download()
        return artifact_path

    @classmethod
    def collect_leaderboard_table(cls, table_name: str, leaderboard_table: wandb.Table):
        cls._instance.leaderboard_tables[table_name] = leaderboard_table

    @classmethod
    def log_overall_leaderboard_table(cls, model_name: str, dataset_names: List[str]) -> wandb.Table:
        final_score_key_dict = {
            "mt_bench": {
                "columns": ["model_name", "roleplay/average_judge_score", "humanities/average_judge_score", "writing/average_judge_score", "reasoning/average_judge_score"],
                "mapper": {
                    "roleplay/average_judge_score": "GLP_표현",
                    "humanities/average_judge_score": "GLP_표현",
                    "writing/average_judge_score": "GLP_표현",
                    "reasoning/average_judge_score": "GLP_논리적추론"
                }
            },
            "hle": 
            {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_전문적지식"
                }
            },
            "hrm8k": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_수학적추론"
                }
            },
            "aime2025": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_수학적추론"
                }
            },
            "kobalt_700": {
                "columns": ["model_name", "Semantics/accuracy", "Syntax/accuracy"],
                "mapper": {
                    "Semantics/accuracy": "GLP_의미해석",
                    "Syntax/accuracy": "GLP_구문해석"
                }
            },
            "kmmlu": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_일반적지식"
                }
            },
            "kmmlu_pro": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_전문적지식"
                }
            },
            "korean_hate_speech": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "ALT_유해성방지"
                }
            },
            "korean_parallel_corpora": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_번역"
                }
            },
            "haerae_bench_v1": {
                "columns": ["model_name", "reading_comprehension/accuracy", "general_knowledge/accuracy", "history/accuracy", "loan_words/accuracy", "rare_words/accuracy", "standard_nomenclature/accuracy"],
                "mapper": {
                    "reading_comprehension/accuracy": "GLP_의미해석",
                    "general_knowledge/accuracy": "GLP_일반적지식",
                    "history/accuracy": "GLP_일반적지식",
                    "loan_words/accuracy": "GLP_일반적지식",
                    "rare_words/accuracy": "GLP_일반적지식",
                    "standard_nomenclature/accuracy": "GLP_일반적지식"
                }
            },
            "ifeval_ko": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "ALT_제어성"
                }
            },
            "squad_kor_v1": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_정보검색"
                }
            },
            "kobbq": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "ALT_편향성방지"
                }
            },
            "komoral": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "ALT_윤리/도덕"
                }
            },
            "arc_agi": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_추상적추론"
                }
            },
            "swe_bench_verified": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_코딩능력"
                }
            },
            "bfcl": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_함수호출"
                }
            },
            "mrcr_2_needles": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "GLP_장문맥이해"
                }
            },
            "halluLens": {
                "columns": ["model_name", "AVG"],
                "mapper": {
                    "AVG": "ALT_환각방지"
                }
            }
        }
        dataset_name = dataset_names[0]
        columns = final_score_key_dict[dataset_name]["columns"]
        table = cls._instance.leaderboard_tables[dataset_name]
        table = table[columns]
        table.set_index('model_name', inplace=True)
        table = table.rename(columns=final_score_key_dict[dataset_name]["mapper"])
        for dataset_name in dataset_names[1:]:
            columns = final_score_key_dict[dataset_name]["columns"]
            new_table = cls._instance.leaderboard_tables[dataset_name]
            new_table = new_table[columns]
            new_table.set_index('model_name', inplace=True)
            new_table = new_table.rename(columns=final_score_key_dict[dataset_name]["mapper"])
            table = pd.merge(table, new_table, left_index=True, right_index=True)
        table.columns = table.columns.str.replace(r'(_[xy]$)|(\.\d+$)', '', regex=True)
        table_mean = table.T.groupby(table.columns).mean().T
        leaderboard_table = wandb.Table(dataframe=table_mean.reset_index())
        cls._instance.run.log({"leaderboard_table": leaderboard_table})
        # return leaderboard_table