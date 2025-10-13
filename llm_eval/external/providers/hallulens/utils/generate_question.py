# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm.contrib.concurrent import thread_map
import jsonlines
import argparse
import random
from llm_eval.external.providers.hallulens.utils import lm
import pandas as pd
from transformers import AutoTokenizer
import os
from llm_eval.external.providers.hallulens.utils.qa_utils import split_doc, sentence_tokenize


####
"""
    NOTE:
        You need to modify generate functions here to yours.
"""
# 

PRECISE_Q_GENERATION_PROMPT = """I would like you to act as a question generator. I will provide reference and you will generate a factual knowledge based question about "{wiki_title}" based on the reference. The specific requirements are as follows:

1. The question can be fully answered based only on the reference material.
2. The question should be objective and not open-ended.
3. The question should be concise.
4. The question should not require additional information to answer.
5. the question's answer should be a word or a phrase.
6. the question should have only one answer.

Reference:
{wiki_document}

Please reply with the question only without any explanation or additional information:
"""

KO_PRECISE_Q_GENERATION_PROMPT = """저는 당신이 질문 생성기로서 역할을 해주기를 바랍니다. 제가 참고 자료를 제공하면, 당신은 그 자료를 바탕으로 "{wiki_title}"에 대한 사실 기반 지식 질문을 생성해야 합니다. 구체적인 요구사항은 다음과 같습니다:
질문은 오직 참고 자료만을 기반으로 완전히 답변할 수 있어야 합니다.

1. 질문은 객관적이어야 하며 개방형 질문이 아니어야 합니다.
2. 질문은 간결해야 합니다.
3. 질문에 답하기 위해 추가적인 정보가 필요하지 않아야 합니다.
4. 질문의 답변은 단어 또는 구(phrase)여야 합니다.
5. 질문에는 단 하나의 정답만 있어야 합니다.

참고 자료:
{wiki_document}

별도의 설명이나 추가 정보 없이 질문만 회신해 주세요:
"""

PRECISE_ANSWERABILITY_PROMPT = """I would like you to judge question's answerability and answer the question. 
I will provide a question and reference document, and you will judge whether the question is fully answerable based only on the reference document, i.e., whether the answer is included in the reference. 
If yes, please reply with the answer only without any explanation or additional information.
If no, please reply with "unanswerable" only.

Reference document: {ref_document}

Question: {question}"""

KO_PRECISE_ANSWERABILITY_PROMPT = """저는 당신이 질문의 답변 가능성을 판단하고 질문에 답변해주기를 바랍니다.
제가 질문과 참고 문서를 제공하면, 당신은 그 질문이 오직 참고 문서만을 기반으로 완전히 답변 가능한지, 즉 정답이 참고 문서에 포함되어 있는지를 판단해야 합니다.
답변이 가능하다면, 어떤 설명이나 추가 정보 없이 정답만 회신해 주세요.
답변이 불가능하다면, "unanswerable"라고만 회신해 주세요.

참고 문서: {ref_document}

질문: {question}"""

LONGFORM_Q_GENERATION_PROMPT ="""I would like you to act as an essay question generator. I will provide a reference and you will generate a factual knowledge based question about "{wiki_title}" based on the reference. The specific requirements are as follows:
1. The question can be fully answered based only on the reference.
2. The question should be objective and not open-ended.
3. The question should be concise.
4. The question's answer should be longer than three sentences.
5. The question should provide enough context to be answered without ambiguity.

Example questions:
Question 1. How did Martin Van Buren become Vice President?
Question 2. What did Neil Armstrong do after retiring from NASA?
Question 3. Describe actions that drive a brownie from Folklore away or cause him to vanish forever.
Question 4. Explain the significance of the Hinomaru Yosegaki in modern times.
Question 5. What are the characteristics and motivations of Datuk Meringgih in the story Sitti Nurbaya?

Reference:
{wiki_document}

Please reply with the question only without any explanation or additional information. 
Remember requirements. Ask only one question. Keep it concise.
If you cannot generate an essay question, please reply with "[NO QUESTION]".
Question: 
"""

KO_LONGFORM_Q_GENERATION_PROMPT = """저는 당신이 서술형 문제 출제자 역할을 해주기를 바랍니다. 제가 참고 자료를 제공하면, 당신은 그 자료에 기반하여 '{wiki_title}'에 대한 사실 기반 질문을 생성해야 합니다. 구체적인 요구사항은 다음과 같습니다:
1. 질문은 제공된 참고 자료만으로 완전히 답변할 수 있어야 합니다.
2. 질문은 객관적이어야 하며, 정해진 답이 없는 질문(open-ended)이 아니어야 합니다.
3. 질문은 간결해야 합니다.
4. 질문에 대한 답변은 세 문장보다 길어야 합니다.
5. 질문은 모호함 없이 답변할 수 있도록 충분한 맥락을 제공해야 합니다.

질문 예시:
질문 1. 마틴 밴 뷰런은 어떻게 부통령이 되었습니까?
질문 2. 닐 암스트롱은 NASA에서 은퇴한 후 무엇을 했습니까?
질문 3. 민담에 나오는 브라우니를 쫓아내거나 영원히 사라지게 하는 행동들을 설명하시오.
질문 4. 현대에 있어 히노마루 요세가키의 중요성을 설명하시오.
질문 5. 소설 <시티 누르바야>에 나오는 다툭 메링기흐의 특징과 동기는 무엇입니까?

참고 자료:
{wiki_document}

어떠한 설명이나 추가 정보 없이 질문만으로 회신해 주세요.
요구사항을 기억하세요. 질문은 오직 하나만 간결하게 해야 합니다.
만약 서술형 질문을 생성할 수 없다면 '[NO QUESTION]'이라고 회신해 주세요.
질문:
"""

LONGFORM_ANSWERABILITY_PROMPT = """I would like you to judge question's answerability based on the reference document.
I will provide a question and reference document, and you will judge whether the question is fully answerable based only on the reference document, i.e., whether the answer is included in the reference. 
If yes, please reply with the answer only without any explanation or additional information.
If no, please reply with "unanswerable" only.

Reference document: {ref_document}

Question: {question}"""

KO_LONGFORM_ANSWERABILITY_PROMPT = """저는 당신이 참고 문서를 바탕으로 질문의 답변 가능성을 판단하기를 바랍니다.
제가 질문과 참고 문서를 제공하면, 당신은 오직 해당 참고 문서에만 근거하여 질문에 완전히 답변할 수 있는지, 즉 답변이 참고 문서에 포함되어 있는지를 판단해야 합니다.
만약 그렇다면, 어떠한 설명이나 추가 정보 없이 답변만 회신해 주세요.
만약 그렇지 않다면, "unanswerable"이라고만 회신해 주세요.

참고 문서: {ref_document}

질문: {question}"""

class WikiQA:
    def __init__(self, q_generator_path, task, language='kor'):
        self.task = task # 'longform' or 'precise'
        assert task in ['longform', 'precise']

        self.q_generator = q_generator_path
        self.Q_FAIL_TIME = 3 if task == 'precise' else 2
        self.min_len = 200 if task == 'precise' else 500
        self.max_len = 500 if task == 'precise' else 750

        # prompt
        if language == 'kor':
            self.Q_GENERATION_PROMPT = KO_PRECISE_Q_GENERATION_PROMPT if task == 'precise' else KO_LONGFORM_Q_GENERATION_PROMPT
            self.ANSWERABILITY_PROMPT = KO_PRECISE_ANSWERABILITY_PROMPT if task == 'precise' else KO_LONGFORM_ANSWERABILITY_PROMPT
        elif language == 'en':
            self.Q_GENERATION_PROMPT = PRECISE_Q_GENERATION_PROMPT if task == 'precise' else LONGFORM_Q_GENERATION_PROMPT
            self.ANSWERABILITY_PROMPT = PRECISE_ANSWERABILITY_PROMPT if task == 'precise' else LONGFORM_ANSWERABILITY_PROMPT
        else:
            raise NotImplementedError

        self.encoding = AutoTokenizer.from_pretrained(q_generator_path, trust_remote_code=True)

    def generate_QA_with_doc(self, title, document, language='en', min_len=500, max_len=750, only_one_doc=False):
        sections = split_doc(document, language, self.encoding, keep_end=False, keep_colon=False, MIN_LEN=min_len, MAX_LEN=max_len)
        if len(sections) > 2:
            sections = sections[:-1] 
            # last section usually is the reference list

        paired_rqas = []

        if only_one_doc:
            sections = random.sample(sections, 1)
            
        for section in sections:
            fail_time = 0
            while fail_time < self.Q_FAIL_TIME:
                q = self.generate_question_with_doc(title, section, language)
                if q == -1: return [] # when the q generation failed
                a = self.generate_answerability(q, section, language)
                
                if a == -1:
                    fail_time += 1
                    # if fail_time == 3: assert False
                    if fail_time == self.Q_FAIL_TIME: continue
                    continue
                else:
                    break
            paired_rqas.append({"reference": section, \
                                "question": q, "answer": a})

        return paired_rqas

    def generate_question_with_doc(self, title, document):
        instruct = self.Q_GENERATION_PROMPT 
        prompt = instruct.format(wiki_title=title, wiki_document=document.strip())
        reply = lm.generate(prompt, self.q_generator, temperature=0.7, top_p=0.9)
        if reply.lower().startswith("unfortunately"):
            return -1
        
        return reply.strip()
    
    def generate_answerability(self, q, doc):
        instruct = self.ANSWERABILITY_PROMPT
        prompt = instruct.format(ref_document=doc, question=q)
        reply = lm.generate(prompt, self.q_generator, temperature=0.3).strip()
        return self.justify_answerability(reply)
    
    def justify_answerability(self, reply):
        if reply.strip().lower() == "unanswerable"\
                or "unanswerable" in reply\
                    or reply.lower().startswith("unfortunately"):
            print(f"DEBUG: Filtered - unanswerable: {reply[:50]}...")
            return -1
        if self.task == 'longform':
            # this is to ensure the question is "longform" answer triggering
            if len(sentence_tokenize(reply, 'en', False, keep_colon=False)) < 4:
                print(f"DEBUG: Filtered - only {len(sentence_tokenize(reply, 'en', False, keep_colon=False))} sentences (need 4+)")
                print(f"  Answer: {reply[:]}...")
                return -1
        elif self.task == 'precise':
            if len(reply.split()) > 10:
                return -1
        return reply.strip()

    def read_existing_file(self, from_scratch, output_path):
        out_lines = []
        if os.path.isfile(output_path):
            if from_scratch:
                with open(output_path, "w") as f:
                    f.write("")
            else:
                with jsonlines.open(output_path) as f:
                    out_lines = list(f)
        print("Already having questions N=", len(out_lines))
        return out_lines
 
    def per_bin_generation_batch(self, wiki_data, output_path, N):
        QAs = []

        output_dir = "/".join(output_path.split("/")[:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. GENERATE QUESTIONS
        # print("Making prompts to generate questions...")
        all_data, Q_MAKING_PROMPTS = [], []
        for line in wiki_data:

            title = line["title"]
            document = line["document"]
            obj = {"title": title, "h_score_cat": line['h_score_cat'],
                    'pageid': line['pageid'], 'revid': line['revid'], 'description': line['description'], 'categories': line['categories']} # meta data

            # select section from the document
            sections = split_doc(document, "en", self.encoding, keep_end=False, keep_colon=False, MIN_LEN=self.min_len, MAX_LEN=self.max_len)
            if len(sections) > 2: sections = sections[:-1] 
            section = random.sample(sections, 1)[0] # always selecting one section
            obj['reference'] = section

            # make prompt
            instruct = self.Q_GENERATION_PROMPT 
            prompt = instruct.format(wiki_title=title, wiki_document=section.strip())
            
            # append prompt and data
            Q_MAKING_PROMPTS.append(prompt)
            all_data.append(obj)

        print("Generating questions...")
        # TODO: I refined this part to use together api
        results = thread_map(lambda p: lm.call_together_api(p, self.q_generator, temperature=0.7, top_p=0.9),
                                Q_MAKING_PROMPTS,
                                max_workers=50,
                                desc=f"using {self.q_generator}")
        for i, r in enumerate(results):
            all_data[i]['prompt'] = r


        # 2. CHECK ANSWERABILITY
        print("Making prompts to check answerability...")
        instruct = self.ANSWERABILITY_PROMPT
        prompts_answerability = [instruct.format(ref_document=obj['reference'], question=obj['prompt']) \
                                    for obj in all_data]
        print("Generating answers...")
        ans_results = thread_map(lambda p: lm.generate(p, self.q_generator),
                                    prompts_answerability,
                                    max_workers=50,
                                    desc=f"using {self.q_generator}")
        filter_count = 0
        print("Filtering out unanswerable questions...")
        for i, answer in enumerate(ans_results):
            answer_justified = self.justify_answerability(answer)
            if answer_justified == -1:
                filter_count += 1
                continue # filter out unanswerable questions
            else:
                all_data[i]['answer'] = answer
                QAs.append(all_data[i])
                with jsonlines.open(output_path, 'a') as writer:
                    writer.write(all_data[i])

            if len(QAs) >= N:
                print("Finished. Filter out {} unanswerable questions.".format(filter_count))
                break
        print(filter_count)
            
        return QAs

############################################################################################################
def precise_QA_generation_run_batch(
        wiki_input_path,
        N=5000,
        q_generator="meta-llama/Meta-Llama-3.1-70B-Instruct",
        output_path="",
        from_scratch=False,
    ):
    print("Wiki Source ={}...".format(wiki_input_path))
    qa = WikiQA(q_generator, task='precise')

    wiki_data_all = pd.read_json(wiki_input_path, orient='records', lines=True)

    # level set up
    low_level, high_level = 0, 10
    per_level_count = N//(high_level-low_level)

    print()
    print("START TO GENERATE QUESTION N={}...".format(N))
    QAs_all = []

    for bin in range(low_level,high_level):

        level_wiki = wiki_data_all[wiki_data_all['h_score_cat'] == bin]
        level_wiki = level_wiki.sample(frac=1) 
        wiki_data = level_wiki.to_dict(orient='records')
        random.shuffle(wiki_data)

        wiki_data = wiki_data[:per_level_count+100] # add 100 buffer
        bin_QAs = qa.per_bin_generation_batch(wiki_data, output_path, per_level_count)
        QAs_all.extend(bin_QAs)

    return QAs_all
############################################################################################################
def longform_QA_generation_run_batch(
        wiki_input_path,
        N=250,
        q_generator="meta-llama/Llama-3.1-70B-Instruct",
        output_path="",
        from_scratch=False,
        low_level=5,
        high_level=10
    ):
    """You need to use a powerful model for longform QA generation to model follow prompt, e.g., Llama-3.1-70B-Instruct"""
    qa = WikiQA(q_generator, task='longform')

    print("START TO GENERATE QUESTION N={}...".format(N))
    print("Wiki Source ={}...".format(wiki_input_path))

    QAs_all = []
    QAs = qa.read_existing_file(from_scratch, output_path)
    if len(QAs) >= N:
        print("Already having questions N={}...".format(len(QAs)))
        return QAs[:N]
    
    wiki_data_all = pd.read_json(wiki_input_path, orient='records', lines=True)
    with open("data/wiki_data/doc_goodwiki_not_exist_titles.txt", 'r') as f:
        not_exist =[f.strip() for f in f.readlines()]
    wiki_data_all = wiki_data_all[~wiki_data_all['title'].isin(not_exist)]

    # level set up
    low_level, high_level = low_level, high_level
    per_level_count = N//(high_level-low_level) if N != 250 else 50

    for bin in range(low_level,high_level):
        level_wiki = wiki_data_all[wiki_data_all['h_score_cat'] == bin]
        level_wiki = level_wiki.sample(frac=1) 
        wiki_data = level_wiki.to_dict(orient='records')
        random.shuffle(wiki_data)
        wiki_data = wiki_data[:per_level_count*5] #todo 100

        bin_QAs = qa.per_bin_generation_batch(wiki_data, output_path, per_level_count)
        assert len(bin_QAs) == per_level_count, f"{len(bin_QAs)} != {per_level_count}"
        QAs_all.extend(bin_QAs)

    return QAs_all

if __name__ == "__main__":
    print("start to generate_question...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="doc_anah.jsonl")
    parser.add_argument("--output_path", type=str, default="qa.jsonl")
    parser.add_argument("--language", type=str, default="kor")
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument("--max_doc_num", type=int, default=-1)
    parser.add_argument("--min_len", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=400)
    args = parser.parse_args()
    


