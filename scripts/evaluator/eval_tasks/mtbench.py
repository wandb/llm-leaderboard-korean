# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import WandbConfigSingleton
from evaluator.eval_utils import (
    task_to_sub_category,
)

import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

from fastchat.conversation import initialize_vllm_custom_template
from fastchat.llm_judge.gen_api_answer import get_api_answer
from fastchat.llm_judge.gen_judgment import *
from fastchat.llm_judge.common import (
        load_questions,
        load_model_answers,
        load_judge_prompts,
        check_data,
        play_a_match_pair,
        play_a_match_single,
        NEED_REF_CATS,
    )
from .abstract import AbstractEvaluator

class MTBenchEvaluator(AbstractEvaluator):
    def __init__(self, few_shots:bool):
        super().__init__(few_shots)
        self.dataset_name = "mtbench"
        self.tasks = ["humanities",
            "roleplay",
            "writing",
            "reasoning",
            "math",
            "extraction",
            "stem"]

    def evaluate(self):
        # create hash and append it to the model_id in order to avoid duplicated id
        mnaum_data = str(datetime.datetime.now())
        encoded_data = mnaum_data.encode()
        hash_object = hashlib.sha256(encoded_data)
        hashed_string = hash_object.hexdigest()
        if self.cfg.mtbench.model_id == None:
            self.cfg.mtbench.model_id = f'{self.cfg.model.pretrained_model_name_or_path.replace("/", "--")}_hash_{hashed_string}' 

        if self.cfg.api == 'vllm':
            initialize_vllm_custom_template()

        if self.cfg.mtbench.num_gpus_total // self.cfg.mtbench.num_gpus_per_model > 1:
            import ray
            ray.init()
        
        ## file path
        #question
        if self.cfg.testmode:
            artifact_dir = self.run.use_artifact(self.cfg.mtbench.question_artifacts_path_test, type='dataset').download()
        else:
            artifact_dir = self.run.use_artifact(self.cfg.mtbench.question_artifacts_path, type='dataset').download()
        question_file = artifact_dir + f"/question.jsonl"
        
        #create answerfile and answerdir
        answer_file = f"fastchat/llm_judge/data/{self.cfg.mtbench.bench_name}/model_answer/{self.cfg.mtbench.model_id}.jsonl"
        answer_dir = f"fastchat/llm_judge/data/{self.cfg.mtbench.bench_name}/model_answer"

        #reference answer
        if self.cfg.testmode:
            ref_answer_dir = self.run.use_artifact(self.cfg.mtbench.referenceanswer_artifacts_path_test, type='dataset').download()
        else:
            ref_answer_dir = self.run.use_artifact(self.cfg.mtbench.referenceanswer_artifacts_path, type='dataset').download()

        
        # 1. generate model answers
        if self.cfg.api in ["openai","anthropic","cohere","google","amazon_bedrock","mistral","upstage","deepseek","vllm","wandb","kakaocorp_private"]:
            questions = load_questions(question_file, None, None)
            get_api_answer(
                question_file=question_file,
                answer_file=answer_file,
                num_worker=self.cfg.mtbench.parallel,
            )

        # 2. evaluate outputs
        questions = load_questions(question_file, None, None)
        model_answers = load_model_answers(answer_dir)
        model_answers = {self.cfg.mtbench.model_id: model_answers[self.cfg.mtbench.model_id]}
        ref_answers = load_model_answers(ref_answer_dir)
        artifact_dir = self.run.use_artifact(self.cfg.mtbench.judge_prompt_artifacts_path, type='dataset').download()
        judge_prompts = load_judge_prompts(artifact_dir + "/judge_ko_prompts.jsonl")

        if self.cfg.mtbench.first_n:
            questions = questions[: self.cfg.mtbench.first_n]

        models = [self.cfg.mtbench.model_id] #get_model_list(answer_dir)
    
        #if self.cfg.mtbench.mode == "single":
        judges = make_judge_single(self.cfg.mtbench.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = f"fastchat/llm_judge/data/{self.cfg.mtbench.bench_name}/model_judgment/{self.cfg.mtbench.judge_model}_single"
        make_match_func = make_match_single
        baseline_model = None

        check_data(questions, model_answers, ref_answers, models, judges)

        question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
        question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

        # Make matches
        matches = []
        matches += make_match_func(
            question_default, models, model_answers, judges["default"], baseline_model
        )
        matches += make_match_func(
            question_math,
            models,
            model_answers,
            judges["math"],
            baseline_model,
            ref_answers,
        )
        matches += make_match_func(
            question_default,
            models,
            model_answers,
            judges["default-mt"],
            baseline_model,
            multi_turn=True,
        )
        matches += make_match_func(
            question_math,
            models,
            model_answers,
            judges["math-mt"],
            baseline_model,
            ref_answers,
            multi_turn=True,
        )

        judge_count = self.cfg.mtbench.get('judge_count', 1)
        matches *= judge_count

        match_stat = {}
        match_stat["bench_name"] = self.cfg.mtbench.bench_name
        match_stat["mode"] = self.cfg.mtbench.mode
        match_stat["judge"] = self.cfg.mtbench.judge_model
        match_stat["baseline"] = baseline_model
        match_stat["model_list"] = models
        match_stat["total_num_questions"] = len(questions)
        match_stat["total_num_matches"] = len(matches)
        match_stat["output_path"] = output_file

        # Show match stats and prompt enter to continue
        print("Stats:")
        print(json.dumps(match_stat, indent=4))
        #input("Press Enter to confirm...")

        # Play matches
        import concurrent.futures
        from tqdm import tqdm
        if self.cfg.mtbench.parallel == 1:
            for match in tqdm.tqdm(matches):
                play_a_match_func(match, output_file=output_file)
        else:
            def play_a_match_wrapper(match):
                play_a_match_func(match, output_file=output_file)

            np.random.seed(0)
            np.random.shuffle(matches)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.mtbench.parallel) as executor:
                futures = [executor.submit(play_a_match_wrapper, match) for match in matches]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Function raised an exception: {e}")

        # 3. consolidate results and log as wandb.Table
        # load questions
        df_question = pd.read_json(question_file, lines=True)
        df_question_repeat = df_question.loc[df_question.index.repeat(judge_count)].reset_index(drop=True)
        # load answers
        df_answer = pd.read_json(answer_file, lines=True)
        df_answer = df_answer[(df_answer.model_id == self.cfg.mtbench.model_id)|(df_answer.model_id == self.cfg.model.pretrained_model_name_or_path)]
        df_answer = df_answer.sort_values(['question_id'])
        df_answer_repeat = df_answer.loc[df_answer.index.repeat(judge_count)].reset_index(drop=True)

        # load judge results
        output_file_turn1 = output_file + "/"+ self.cfg.mtbench.model_id + "__1turn.jsonl"
        output_file_turn2 = output_file + "/"+ self.cfg.mtbench.model_id + "__2turn.jsonl"
        print(output_file_turn1)
        # with open(output_file_turn1, 'r') as f:
        #     data = [json.loads(line) for line in f]
        # df_judge1 = pd.DataFrame(data)
        # with open(output_file_turn2, 'r') as f:
        #     data = [json.loads(line) for line in f]
        # df_judge2 = pd.DataFrame(data)
        df_judge1 = pd.read_json(output_file_turn1, lines=True)
        df_judge2 = pd.read_json(output_file_turn2, lines=True)
        df_judge = pd.concat([df_judge1, df_judge2], ignore_index=True)

        df_judge = df_judge[df_judge.model == self.cfg.mtbench.model_id]
        df_judge.model = df_judge.model.str.replace("--", "/")
        df_judge['hash'] = df_judge.model.apply(lambda x: x.split('_hash_')[-1])
        df_judge['model'] = df_judge.model.apply(lambda x: x.split('_hash_')[0])
        df_judge = df_judge.sort_values(['question_id', 'turn'])

        ## merge tables
        df_judge["question"] = pd.Series(np.nan, dtype='object')
        df_judge.loc[df_judge.turn == 1, 'question'] = df_question_repeat.turns.apply(lambda x: x[0]).values
        df_judge.loc[df_judge.turn == 2, 'question'] = df_question_repeat.turns.apply(lambda x: x[1]).values

        df_judge['answer'] = pd.Series(np.nan, dtype='object')
        df_judge.loc[df_judge.turn == 1, 'answer'] = df_answer_repeat.choices.apply(lambda x: x[0]['turns'][0]).values
        df_judge.loc[df_judge.turn == 2, 'answer'] = df_answer_repeat.choices.apply(lambda x: x[0]['turns'][1]).values
        df_judge = df_judge.merge(df_answer[['question_id', 'answer_id']], on='question_id', how='left')
        df_judge = df_judge.merge(df_question[['question_id', 'category']], on='question_id', how='left')

        ## clean dataframe up
        use_col = [
            'model','question_id', 'category', 'question', 
            'answer', 'judge', 'user_prompt', 'judgment', 
            'score', 'turn', 'tstamp'
        ]
        df_judge = df_judge[use_col]
        df_judge.rename(columns={'model': 'model_name'}, inplace=True)
        df_judge['sub_category'] = df_judge['category'].map(task_to_sub_category)

        table_log = wandb.Table(dataframe=df_judge)
        

        # table for radar chart
        _df_judge = df_judge.query('score != -1').groupby(['question_id', 'turn', 'category'], as_index=False).score.mean()
        df_summary = _df_judge.groupby(['category'], as_index=False).score.mean()
        table_radar = wandb.Table(dataframe=df_summary)

        ## table for LB mtbench
        mtbench_df = pd.DataFrame([df_summary.score.values.tolist()], columns=df_summary.category.values.tolist())
        mtbench_df.insert(0, "AVG_mtbench",  mtbench_df.mean(axis=1, numeric_only=True))
        mtbench_df.insert(0, "model_name",  self.cfg.model.pretrained_model_name_or_path)
        table_metric = wandb.Table(dataframe=mtbench_df)

        self.run.log({
            "mtbench_output_table":table_log,
            "mtbench_leaderboard_table":table_metric,
            "mtbench_radar_table":table_radar,
        })

        return
