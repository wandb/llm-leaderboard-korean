from utils import read_wandb_table, WandbConfigSingleton

from pathlib import Path
import os
import wandb
import pandas as pd
import numpy as np

def radar_contents(leaderboard_dict, categories: list[str]) -> list[list[str, float]]:
    ret = []
    for cat in categories:
        ret.append([cat[4:], leaderboard_dict[cat]])
    return ret

def update_flag(cfg, blend_cfg):
    mtbench_flag = kobbq_flag = kaster_flag = GLP_flag = ALT_flag = False

    if hasattr(cfg, 'run'):
        mtbench_flag = cfg.run.mtbench
        kobbq_flag = cfg.run.kobbq
        kaster_flag = cfg.run.kaster
        # ko_truthful_qa_flag = cfg.run.ko_truthful_qa

    if blend_cfg:
        for old_run in blend_cfg.old_runs:
            print(old_run.dataset)
            if old_run.dataset is None:
                continue
            for dataset in old_run.dataset:
                if "mtbench" in dataset:
                    mtbench_flag = True
                elif "kobbq" in dataset:
                    kobbq_flag = True
                elif "ko_truthful_qa" in dataset:
                    ko_truthful_qa_flag = True
                elif "kaster" in dataset:
                    kaster_flag = True

    if mtbench_flag and kaster_flag:
        GLP_flag = True
    # if kobbq_flag and ko_truthful_qa_flag:
    #     ALT_flag = True
    ALT_flag = True
    return GLP_flag, ALT_flag


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    blend_cfg = instance.blend_config
    num_few_shots = cfg.num_few_shots

    GLP_flag, ALT_flag = update_flag(cfg, blend_cfg)

    # Initialize empty variables
    if GLP_flag or ALT_flag:
        kaster_0shot = kaster_fewshots = kmmlu_robust_fewshots = kaster_control_0shot = None
        kaster_control_fewshots = kobbq_fewshots = toxicity = mtbench = None
        kaster_0shot = read_wandb_table(table_name=f"kaster_0shot_leaderboard_table", run=run)
        kaster_fewshots = read_wandb_table(table_name=f"kaster_{num_few_shots}shot_leaderboard_table", run=run)

    if GLP_flag:
        mtbench = read_wandb_table(table_name=f"mtbench_leaderboard_table", run=run)
        haerae_bench_v1_0shot = read_wandb_table(table_name=f"haerae_bench_v1_0shot_leaderboard_table", run=run)
        haerae_bench_v1_fewshots = read_wandb_table(table_name=f"haerae_bench_v1_{num_few_shots}shot_leaderboard_table", run=run)
        kaster_0shot = pd.concat([kaster_0shot, haerae_bench_v1_0shot], axis=1)
        kaster_fewshots = pd.concat([kaster_fewshots, haerae_bench_v1_fewshots], axis=1)
    
    if ALT_flag:

        kmmlu_robust_fewshots = read_wandb_table(table_name=f"kmmlu_robust_{num_few_shots}shot_leaderboard_table", run=run)
        kaster_control_0shot = read_wandb_table(table_name=f"kaster_control_0shot_leaderboard_table", run=run)
        kaster_control_fewshots = read_wandb_table(table_name=f"kaster_control_{num_few_shots}shot_leaderboard_table", run=run)
        haerae_bench_v1_control_0shot = read_wandb_table(table_name=f"haerae_bench_v1_control_0shot_leaderboard_table", run=run)
        haerae_bench_v1_control_fewshots = read_wandb_table(table_name=f"haerae_bench_v1_control_{num_few_shots}shot_leaderboard_table", run=run)
        kaster_control_0shot = pd.concat([
            kaster_control_0shot.drop(columns=["model_name", "AVG"]),
            haerae_bench_v1_control_0shot.drop(columns=["model_name", "AVG"])],
            axis=1)
        kaster_control_0shot.insert(0, 'AVG', kaster_control_0shot.mean(axis=1))
        kaster_control_0shot.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
        kaster_control_fewshots = pd.concat([
            kaster_control_fewshots.drop(columns=["model_name", "AVG"]),
            haerae_bench_v1_control_fewshots.drop(columns=["model_name", "AVG"])],
            axis=1)
        kaster_control_fewshots.insert(0, 'AVG', kaster_control_fewshots.mean(axis=1))
        kaster_control_fewshots.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
        
        kobbq_fewshots = read_wandb_table(table_name=f"kobbq_{num_few_shots}shot_leaderboard_table", run=run)
        #TODO
        # toxicity = read_wandb_table(table_name=f"toxicity_leaderboard_table", run=run)
        ko_truthful_qa = read_wandb_table(table_name=f"ko_truthful_qa_0shot_leaderboard_table", run=run)

    print("-------- aggregating results ----------")

    def calculate_combined_means(cols_kaster, cols_mtbench):
        means = []
        if cols_kaster:
            for col in cols_kaster:
                mean_value = (kaster_0shot[col][0] + kaster_fewshots[col][0]) / 2
                means.append(mean_value)

        if cols_mtbench:
            for col in cols_mtbench:
                means.append(mtbench[col][0] / 10)
        return np.mean(means)

    def create_subcategory_table(category, cols_kaster, cols_mtbench, other=None):
        table_name = f"subcategory_table_{category}"
        data = {}

        if other is None:
            data["model_name"]=cfg.model.pretrained_model_name_or_path
            data["AVG"] = calculate_combined_means(cols_kaster, cols_mtbench)
            if cols_kaster:
                for col in cols_kaster:
                    data[f"{col}_0shot"] =  kaster_0shot[col][0]
                    data[f"{col}_{num_few_shots}shot"] = kaster_fewshots[col][0]
            if cols_mtbench:
                for col in cols_mtbench:
                    data[f"{col}_mtbench"] = mtbench[col][0] / 10
        
        elif other == "control":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": np.mean([np.mean([kaster_control_0shot["AVG"][0], kaster_control_fewshots["AVG"][0]])]),
                "kaster_control_0shot":kaster_control_0shot["AVG"][0],
                "kaster_control_2shot":kaster_control_fewshots["AVG"][0],
            }

        elif other == "toxicity":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": np.mean([np.mean([kaster_0shot["korean-hate-speech_hate"][0], kaster_fewshots["korean-hate-speech_hate"][0]])]),
                "korean-hate-speech_hate_0shot": kaster_0shot["korean-hate-speech_hate"][0],
                "korean-hate-speech_hate_2shot": kaster_fewshots["korean-hate-speech_hate"][0],
            }
            # data = {
            #     "model_name": cfg.model.pretrained_model_name_or_path,
            #     "AVG": toxicity[["公平性", "社会規範", "禁止行為", "違反カテゴリ"]].values.mean(),
            #     "公平性": toxicity["公平性"][0],
            #     "社会規範": toxicity["社会規範"][0],
            #     "禁止行為": toxicity["禁止行為"][0],
            #     "違反カテゴリ": toxicity["違反カテゴリ"][0],
            # }

        elif other == "bias":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": np.mean([np.mean([
                    # kaster_0shot["korean-hate-speech_bias"][0],
                    # kaster_fewshots["korean-hate-speech_bias"][0],
                    kobbq_fewshots["avg"][0],
                    # kobbq_fewshots["acc_a"][0],
                    # kobbq_fewshots["acc_d"][0],
                    # kobbq_fewshots["diff_bias_a"][0],
                    # kobbq_fewshots["diff_bias_d"][0],
                    ])]),
                # "korean-hate-speech_bias_0shot": kaster_0shot["korean-hate-speech_bias"][0],
                # "korean-hate-speech_bias_2shot": kaster_fewshots["korean-hate-speech_bias"][0],
                "kobbq_ACC_amb_2shot": kobbq_fewshots["acc_a"][0],
                "kobbq_ACC_disamb_2shot": kobbq_fewshots["acc_d"][0],
                "kobbq_DIFF_bias_amb_2shot": kobbq_fewshots["diff_bias_a"][0],
                "kobbq_DIFF_bias_disamb_2shot": kobbq_fewshots["diff_bias_d"][0],
            }
            # data = {
            #     "model_name": cfg.model.pretrained_model_name_or_path,
            #     "AVG": 1 - kobbq_fewshots["avg_abs_bias_score"][0],
            #     "abs_bias_score_fewshot": kobbq_fewshots["avg_abs_bias_score"][0],
            # }

        elif other == "robust":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": kmmlu_robust_fewshots["robust_score"][0],
                "kmmlu_robust_fewshots": kmmlu_robust_fewshots["robust_score"][0],
            }

        elif other == "truthful":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": ko_truthful_qa["ko_truthful_qa-generation"][0],
                "ko_truthful_qa_overall_score": ko_truthful_qa["ko_truthful_qa-generation"][0],
            }

        # Convert data to DataFrame
        subcategory_table = pd.DataFrame([data])
        run.log({table_name: wandb.Table(dataframe=subcategory_table)})

    def calculate_average_from_dict(data_dict, prefix):
        relevant_items = {key: value for key, value in data_dict.items() if key.startswith(prefix)}
        relevant_values = [value for value in relevant_items.values() if isinstance(value, (int, float))]
        if relevant_values:
            return sum(relevant_values) / len(relevant_values)
        return float('nan')

    leaderboard_dict = {}
    leaderboard_dict["model_name"] = cfg.model.pretrained_model_name_or_path
    leaderboard_dict["model_size_category"] = cfg.model.get("size_category", np.nan)
    leaderboard_dict["model_size"] = cfg.model.get("size", np.nan)
    leaderboard_dict["model_release_date"] = pd.to_datetime(cfg.model.release_date, format='%m/%d/%Y')
    first_cols = ["model_name","model_size_category"]
    
    if GLP_flag:
        leaderboard_dict["GLP_표현"] = calculate_combined_means([],["roleplay","writing","humanities"])
        create_subcategory_table("expression", [], ["roleplay","writing","humanities"])
        leaderboard_dict["GLP_번역"] = calculate_combined_means(["korean-parallel-corpora-e2k", "korean-parallel-corpora-k2e"], [])
        create_subcategory_table("translation", ["korean-parallel-corpora-e2k", "korean-parallel-corpora-k2e"], [])
        leaderboard_dict["GLP_정보검색"] = calculate_combined_means(["squad_kor_v1"], [])
        create_subcategory_table("information_extraction", ["squad_kor_v1"], [])
        leaderboard_dict["GLP_추론"] = calculate_combined_means([], ["reasoning"])
        create_subcategory_table("reasoning", [], ["reasoning"])
        leaderboard_dict["GLP_수학적추론"] = calculate_combined_means(["gsm8k"], ["math"])
        create_subcategory_table("mathematical_reasoning", ["gsm8k"], ["math"])
        leaderboard_dict["GLP_추출"] = calculate_combined_means(["klue_ner"], ["extraction"])
        create_subcategory_table("entity_extraction", ["klue_ner"], ["extraction"])
        leaderboard_dict["GLP_지식・질의응답"] = calculate_combined_means(["kmmlu", "haerae_bench-HI", "haerae_bench-KGK", "haerae_bench-LW", "haerae_bench-RW", "haerae_bench-SN"], ["stem"])
        create_subcategory_table("knowledge_QA", ["kmmlu", "haerae_bench-HI", "haerae_bench-KGK", "haerae_bench-LW", "haerae_bench-RW", "haerae_bench-SN"], ["stem"])
        leaderboard_dict["GLP_영어"] = calculate_combined_means(["mmlu_en"], [])
        create_subcategory_table("english", ["mmlu_en"], [])
        leaderboard_dict["GLP_의미해석"] = calculate_combined_means(["kobest_sn", "kornli", "kobest_wic", "haerae_bench-RC"], [])
        create_subcategory_table("semantic_analysis", ["kobest_sn", "kornli", "kobest_wic", "haerae_bench-RC"], [])
        leaderboard_dict["GLP_구문해석"] = calculate_combined_means(["klue_re"], [])   
        create_subcategory_table("syntactic_analysis", ["klue_re"], []) 
        leaderboard_dict["범용적언어성능(GLP)_AVG"] = calculate_average_from_dict(leaderboard_dict, "GLP")
        first_cols.append("범용적언어성능(GLP)_AVG")

    if ALT_flag:
        leaderboard_dict["ALT_제어성"] = np.mean([np.mean([kaster_control_0shot["AVG"][0], kaster_control_fewshots["AVG"][0]])])#, lctg_overall["AVG_Total_ctg"][0]])
        create_subcategory_table("controllability", [], [], "control")
        leaderboard_dict["ALT_윤리・도덕"] = kaster_fewshots["komoral"][0] # use only fewshots result
        create_subcategory_table("ethics", ["komoral"], [])
        leaderboard_dict["ALT_독성"] = kaster_fewshots["korean-hate-speech_hate"][0]
        create_subcategory_table("toxicity", ["korean-hate-speech_hate"], [])
        leaderboard_dict["ALT_사회적편견"] = kobbq_fewshots["avg"][0]
        create_subcategory_table("bias", [], [], "bias")
        leaderboard_dict["ALT_모델강건성"] = kmmlu_robust_fewshots["robust_score"][0]
        create_subcategory_table("robustness", [], [], "robust")
        leaderboard_dict["ALT_진실성"] = ko_truthful_qa["ko_truthful_qa-generation"][0]
        create_subcategory_table("truthfulness", [], [], "truthful")
        leaderboard_dict["Alignment(ALT)_AVG"] = calculate_average_from_dict(leaderboard_dict, "ALT")
        first_cols.append("Alignment(ALT)_AVG")

    if GLP_flag and ALT_flag:
        leaderboard_dict["TOTAL_AVG"] = np.mean([leaderboard_dict["범용적언어성능(GLP)_AVG"], leaderboard_dict["Alignment(ALT)_AVG"]])
        first_cols.append("TOTAL_AVG")

    # Average of each dataset
    if GLP_flag or ALT_flag:
        kaster_agg_cols = [c for c in kaster_0shot if not c.startswith("kmmlu_") and c not in ["run_name", "model_name"]]
        leaderboard_dict["AVG_kaster_0shot"] = kaster_0shot[kaster_agg_cols].mean(axis=1)[0]
        leaderboard_dict[f"AVG_kaster_{num_few_shots}shots"] = kaster_fewshots[kaster_agg_cols].mean(axis=1)[0]
    
    if GLP_flag:
        leaderboard_dict["AVG_mtbench"] = mtbench["AVG_mtbench"][0]
    
    # if ALT_flag:
    #     leaderboard_dict["AVG_lctg"] = lctg_overall["AVG_Total_ctg"][0]

    leaderboard_table = pd.DataFrame([leaderboard_dict])
    cols = leaderboard_table.columns
    new_cols = first_cols + [c for c in cols if c not in first_cols]
    leaderboard_table = leaderboard_table[new_cols]
    # Radar table

    glp_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=leaderboard_dict,
            categories=[
                "GLP_정보검색",
                "GLP_추론",
                "GLP_수학적추론",
                "GLP_추출",
                "GLP_지식・질의응답",
                "GLP_영어",
                "GLP_의미해석",
                "GLP_구문해석",
            ],
        ),
        columns=["category", "score"],
    ) if GLP_flag else pd.DataFrame()

    alt_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=leaderboard_dict,
            categories=[
                "ALT_제어성",
                "ALT_윤리・도덕",
                "ALT_독성",
                "ALT_사회적편견",
                "ALT_모델강건성",
                "ALT_진실성",
            ],
        ),
        columns=["category", "score"],
    ) if ALT_flag else pd.DataFrame()

    run.log({
        "leaderboard_table": wandb.Table(dataframe=leaderboard_table),
        "glp_radar_table": wandb.Table(dataframe=glp_radar_table) if GLP_flag else None,
        "alt_radar_table": wandb.Table(dataframe=alt_radar_table) if ALT_flag else None
    })
    run.finish()