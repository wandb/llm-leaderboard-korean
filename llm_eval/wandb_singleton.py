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
        if "mtbench" in dataset_name:
            artifact = api.artifact(f"wandb-korea/korean-llm-leaderboard/{dataset_name}:latest")
            artifact_path = artifact.download()
            return artifact_path
        artifact = api.artifact(f"{cls._instance.wandb_params.get('entity')}/{cls._instance.wandb_params.get('project_dataset')}/{dataset_name}:latest")
        artifact_path = artifact.download()
        return artifact_path

    @classmethod
    def collect_leaderboard_table(cls, table_name: str, leaderboard_table):
        """
        Collect a leaderboard table for later logging at finish().
        Accepts either a pandas DataFrame or a wandb.Table.
        """
        cls._instance.leaderboard_tables[table_name] = leaderboard_table

    @classmethod
    def log_overall_leaderboard_table(cls, model_name: str, release_date: str, size_category: str, model_size: str, dataset_names: List[str]) -> wandb.Table:
        final_score_key_dict = {
            "mt_bench": {
                "columns": ["model_name", "roleplay", "humanities", "writing", "reasoning", "coding"],
                "mapper": {
                    "roleplay": "GLP_표현",
                    "humanities": "GLP_표현",
                    "writing": "GLP_표현",
                    "reasoning": "GLP_논리적추론",
                    "coding": "GLP_코딩능력"
                }
            },
            "hle": 
            {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_전문적지식"
                }
            },
            "hrm8k": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_수학적추론"
                }
            },
            "aime2025": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_수학적추론"
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
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_일반적지식"
                }
            },
            "kmmlu_pro": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_전문적지식"
                }
            },
            "korean_hate_speech": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "ALT_유해성방지"
                }
            },
            "korean_parallel_corpora": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_번역"
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
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "ALT_제어성"
                }
            },
            "squad_kor_v1": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_정보검색"
                }
            },
            "kobbq": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "ALT_편향성방지"
                }
            },
            "komoral": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "ALT_윤리/도덕"
                }
            },
            "arc_agi": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_추상적추론"
                }
            },
            "swebench": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_코딩능력"
                }
            },
            "bfcl": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_함수호출"
                }
            },
            "mrcr_2_needles": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "GLP_장문맥이해"
                }
            },
            "halluLens": {
                "columns": ["model_name", "score"],
                "mapper": {
                    "score": "ALT_환각방지"
                }
            }
        }
        GLP_COLUMN_WEIGHT = {
            "GLP_구문해석": 1,
            "GLP_의미해석": 1,
            "GLP_표현": 1,
            "GLP_번역": 1,
            "GLP_정보검색": 1,
            "GLP_장문맥이해": 1,
            "GLP_일반적지식": 2,
            "GLP_전문적지식": 2,
            "GLP_수학적추론": 2,
            "GLP_논리적추론": 2,
            "GLP_추상적추론": 2,
            "GLP_함수호출": 2,
            "GLP_코딩능력": 2,
        }
        ALT_COLUMN_WEIGHT = {
            "ALT_제어성": 1,
            "ALT_유해성방지": 1,
            "ALT_편향성방지": 1,
            "ALT_윤리/도덕": 1,
            "ALT_환각방지": 1,
        }
        def weighted_average(df, weights_dict):
            cols = [c for c in weights_dict.keys() if c in df.columns]
            weights = [weights_dict[c] for c in cols]
            return (df[cols].mul(weights, axis=1).sum(axis=1)) / sum(weights)
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

        # Only compute GLP/ALT averages if the required columns exist
        # This handles cases where only external benchmarks are run

        table_mean['범용언어성능(GLP)_AVG'] = weighted_average(table_mean, GLP_COLUMN_WEIGHT)
        table_mean['가치정렬성능(ALT)_AVG'] = weighted_average(table_mean, ALT_COLUMN_WEIGHT)

        # Only compute TOTAL_AVG if both GLP and ALT exist
        table_mean['TOTAL_AVG'] = (table_mean['범용언어성능(GLP)_AVG'] + table_mean['가치정렬성능(ALT)_AVG']) / 2
        table_mean['release_date'] = release_date
        table_mean['size_category'] = 'None' if size_category is None else size_category
        table_mean['model_size'] = 'None' if model_size is None else model_size
        table_mean = table_mean.reset_index()
        # Build desired column list, but only include columns that actually exist
        # This handles cases where external benchmarks don't have GLP/ALT columns
        desired_columns = ['model_name', 'release_date', 'size_category', 'TOTAL_AVG', '범용언어성능(GLP)_AVG', '가치정렬성능(ALT)_AVG'] + list(GLP_COLUMN_WEIGHT.keys()) + list(ALT_COLUMN_WEIGHT.keys())
        existing_columns = [col for col in desired_columns if col in table_mean.columns]
        table_mean = table_mean[existing_columns]
        table_mean.to_csv(f"leaderboard_table.csv", index=False)

        leaderboard_table = wandb.Table(dataframe=table_mean)
        cls._instance.run.log({"leaderboard_table": leaderboard_table})


        df_alt = cls.create_radar_chart(table_mean, ALT_COLUMN_WEIGHT.keys())
        df_glp = cls.create_radar_chart(table_mean, GLP_COLUMN_WEIGHT.keys())
        cls._instance.run.log({"alt_radar_table": wandb.Table(dataframe=df_alt)})
        cls._instance.run.log({"glp_radar_table": wandb.Table(dataframe=df_glp)})
        # return leaderboard_table

    @classmethod
    def create_radar_chart(cls, df: pd.DataFrame, columns: List[str]):
        return df[columns].transpose().rename(columns={0: 'score'})

    @classmethod
    def finish(
        cls,
        summary: Optional[dict] = None,
        log_collected_tables: bool = True,
        reset: bool = True,
    ) -> None:
        """
        Finalize the global W&B run held by the singleton.

        - Logs any collected leaderboard tables (DataFrame or wandb.Table)
        - Optionally updates run summary with provided dict
        - Finishes the run and optionally resets the singleton so it can be re-initialized
        """
        if cls._instance is None:
            return

        inst = cls._instance
        try:
            # Log collected leaderboard tables if any
            if log_collected_tables and getattr(inst, "leaderboard_tables", None):
                for name, table in inst.leaderboard_tables.items():
                    try:
                        if isinstance(table, pd.DataFrame):
                            wb_table = wandb.Table(dataframe=table)
                        else:
                            wb_table = table
                        inst.run.log({f"{name}_leaderboard_table": wb_table})
                    except Exception:
                        # Best-effort logging; continue on error
                        pass

            # Update summary if provided
            if summary:
                try:
                    inst.run.summary.update(summary)
                except Exception:
                    pass

            # Finish the run
            try:
                inst.run.finish()
            except Exception:
                try:
                    wandb.finish()
                except Exception:
                    pass
        finally:
            if reset:
                cls._instance = None