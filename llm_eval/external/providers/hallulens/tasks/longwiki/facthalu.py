# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import json
import tiktoken

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
from tqdm import tqdm

from segtok.segmenter import split_single

from transformers import AutoTokenizer
from llm_eval.external.providers.hallulens.tasks.longwiki.longwiki_retrieval import LongWikiRetrieval, LongWikiDB
import llm_eval.external.providers.hallulens.tasks.longwiki.longwiki_utils as utils


@dataclass
class Claim:
    claim: str
    sentence: object
    refernce: Optional[str] = None
    topic: Optional[str] = None
    search_results: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    is_supported: Optional[bool] = None
    question: Optional[str] = None  # same as generation.prompt


@dataclass
class Sentence:
    sentence: str
    generation: object
    prompt: Optional[str]
    claims: Optional[List[Claim]] = None


@dataclass
class Generation:
    generation: str
    prompt: str
    sentences: Optional[List[Sentence]] = None
    abstain: Optional[bool] = None
    reference: Optional[str] = None
    topic: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.generation + self.prompt)

    def __eq__(self, other) -> bool:
        return self.generation == other.generation and self.prompt == other.prompt


encoding = None#AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-405B-Instruct-FP8', trust_remote_code=True)


class FactHalu:
    def __init__(
            self,
            generations_file_path,  #: str | Path
            output_csv: str,
            abstain_evaluator: str = "meta-llama/Llama-3.1-70B-Instruct",
            refusal_evaluator: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
            claim_extractor: str = "meta-llama/Llama-3.1-405B-Instruct-FP8",
            verifier: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
            k: int = 32,
            eval_cache_path='/data/facthalu_longform/.cache',
            db_path="/data/wiki_data/.cache/enwiki-20230401.db",
            language: str = "kor",
            encoding_model: str = 'meta-llama/Llama-3.1-405B-Instruct-FP8',
            do_extract_only=False
    ):

        self.do_extract_only = do_extract_only
        self.generations_file_path = generations_file_path
        self.output_csv = output_csv
        self.prepare_path()

        self.abstain_evaluator = abstain_evaluator
        self.refusal_evaluator = refusal_evaluator
        self.claim_extractor = claim_extractor
        self.ref_src = "retrieval_relevant"
        self.verifier = verifier

        self.k = k

        self.generations = None
        self.db = LongWikiDB(db_path=db_path)

        self.CACHE_BASE_PATH = eval_cache_path  # used for retrieval cache
        self.embedded_cache_path = f"{self.CACHE_BASE_PATH}/embedding/embed_cache_all.pkl"
        if not os.path.exists(f"{self.CACHE_BASE_PATH}/embedding/"):
            os.makedirs(f"{self.CACHE_BASE_PATH}/embedding/")

        print("Cache path:", self.embedded_cache_path)

        self.language = language
        if self.language == "kor":
            from llm_eval.external.providers.hallulens.tasks.longwiki import ko_prompt_templates
            self.prompt_templates = ko_prompt_templates
        elif self.language == "eng":
            from llm_eval.external.providers.hallulens.tasks.longwiki import prompt_templates
            self.prompt_templates = prompt_templates

    def prepare_path(self):
        self.refusal_path = str(self.output_csv).replace(".csv", "_abstain.jsonl")
        self.extracted_claims_path = str(self.output_csv).replace(".csv", "_all_claims.jsonl")
        self.parsed_claims_path = str(self.output_csv).replace(".csv", "_all_parsed_claims.jsonl")
        self.verification_path = str(self.output_csv).replace(".csv", "_verification_results.jsonl")

    def run(self):
        """
        Evaluate longwiki from model error.
        Saves results to output_csv as jsonl with one line per prompt.
        """
        # check if result output_csv exists
        if os.path.exists(self.output_csv):
            print(f"Output file {self.output_csv} already exists. Exiting.\n")
            final_results_df = pd.read_csv(self.output_csv)
            final_results_score_dict = utils.print_all_metrics(final_results_df, k=self.k)
            return final_results_score_dict, final_results_df

        now = time.time()

        ### [[STEP #0]] Load generations
        self.load_generations()

        ### [[STEP #1]] False Refusal Test
        print("\n[[Step 1]] False Refusal Test starts")
        no_abstains = self.eval_abstention()

        ### [[STEP #2]] Extract claims
        print("\n[[Step 2]] Extracting Claims starts")
        all_claims = self.extract_claims(no_abstains)

        if self.do_extract_only:
            print("Extract only mode. Exiting.")
            return

        # if there is no claim for a generation, mark it as abstain
        no_abstains = [generation for generation in self.generations if not generation.abstain]

        ### [[STEP #3]] Verify claims
        print(f"\n[[Step 3]] Verifying Claims starts. Len: {len(all_claims)}")
        all_verification_responses = self.verify_claims(all_claims)

        for claim, verification_response in zip(all_claims, all_verification_responses):
            claim.is_supported = verification_response["is_supported"]

            ### [[[ STEP #4]]] Calculate metrics: precision, recall@k, f1, response ratio
        print(f"[[Step 4]] Calculating metrics")
        final_results = []
        for generation in no_abstains:
            for sentence in generation.sentences:
                if not sentence.claims:
                    final_results.append(
                        {
                            "prompt": generation.prompt,
                            "is_supported": None,
                            "claim": "no claims",
                            "sentence": sentence.sentence,
                            "title": generation.topic
                        }
                    )
                else:
                    for claim in sentence.claims:
                        final_results.append(
                            {
                                "prompt": generation.prompt,
                                "is_supported": claim.is_supported,
                                "claim": claim.claim,
                                "sentence": sentence.sentence,
                                "title": generation.topic

                            }
                        )
        print("---------------------------------")
        print("Abstain ratio:", "%.3f" % (1 - len(no_abstains) / len(self.generations)))
        final_results_df = pd.DataFrame(final_results)
        final_results_df = utils.calculate_all_metrics(final_results_df, k=self.k)
        final_results_df.to_csv(self.output_csv, index=False)
        print("---------------------------------")
        print(f"Saved detailed results in {self.output_csv}")
        print(f"Took {time.time() - now} seconds")
        print("---------------------------------")

        return utils.print_all_metrics(final_results_df, k=self.k), final_results_df

    ##########################################################################################
    ##########################################################################################
    def load_generations(self):
        self.generations = []
        with open(self.generations_file_path, "r") as f:
            for line in f:
                l = json.loads(line)
                generation = Generation(
                    generation=l["generation"],
                    prompt=l["prompt"],
                )
                # Adding reference article here to replace search
                if self.ref_src == "default":
                    if l.get("reference", None) != None:
                        generation.reference = l["reference"]
                generation.topic = l["title"]
                self.generations.append(generation)

    def eval_abstention(self):
        refusal_path = self.refusal_path

        abstain_prompts = [
            self.prompt_templates.ABSTAIN_PROMPT.format(
                prompt=generation.prompt.strip(), generation=generation.generation
            ).strip()
            for generation in self.generations
        ]

        abstains_eval_raw = utils.read_eval_raw(refusal_path)
        if len(abstains_eval_raw) == len(abstain_prompts):
            print("Read from cache {}".format(refusal_path))
        else:
            abstains_eval_raw = utils.model_eval_step(self.refusal_evaluator, abstain_prompts, max_token=128,
                                                      batch_size=64)
            utils.save_eval_raw(abstains_eval_raw, output_file=refusal_path)

        abstains_eval = utils.jsonify_ans(
            raw_responses=abstains_eval_raw,
            eval_prompts=abstain_prompts,
            evaluator=self.refusal_evaluator,
            key="is_knowledgeable"
        )

        for generation, abstain in zip(self.generations, abstains_eval):
            generation.abstain = not abstain["is_knowledgeable"]

        no_abstains = [generation for generation in self.generations if not generation.abstain]
        return no_abstains

    def extract_claims(self, no_abstains: List[Generation]):
        extracted_claims_path = self.extracted_claims_path
        all_claim_extractions = []

        all_sentences = [
            sentence for g in no_abstains \
            for sentence in
            make_claim_extraction_prompts(g, claim_extractor=self.claim_extractor, language=self.language)
        ]

        all_claim_extractions = utils.read_eval_raw(extracted_claims_path)
        if len(all_claim_extractions) == len(all_sentences):
            print("***** [2-1] Reading extracted claims from cache")
        else:
            to_extract_prompts = [a.prompt for a in all_sentences]
            if all_claim_extractions != []:
                print(f"***** [2-1] Resuming extraction from cache, starting from {len(all_claim_extractions)}\n")
                to_extract_prompts = to_extract_prompts[len(all_claim_extractions):]

            mini_bsz = 100
            for i in range(0, len(to_extract_prompts), mini_bsz):
                batch_prompts = to_extract_prompts[i:i + mini_bsz]
                batch_results = utils.model_eval_step(self.claim_extractor, batch_prompts, max_token=512, batch_size=8,
                                                      max_workers=16)
                all_claim_extractions.extend(batch_results)
                utils.save_eval_raw(all_claim_extractions, output_file=extracted_claims_path)
                if i % 500 == 0: print(f"Processed {i + 100} sentences. out of {len(all_sentences)}")

            utils.save_eval_raw(all_claim_extractions, output_file=extracted_claims_path)

        print("***** [2-2] Parsing extracted claims")
        all_claims = []
        deduplicate = defaultdict(set)
        assert len(all_claim_extractions) == len(all_sentences), ("Mismatch in extracted claims and sentences\n"
                                                                  "{} vs {}".format(len(all_claim_extractions),
                                                                                    len(all_sentences)))

        for claim_extraction, sentence in zip(all_claim_extractions, all_sentences):
            if (not claim_extraction) or \
                    claim_extraction.strip() == "No verifiable claim." or \
                    claim_extraction.strip() == "No available facts" or \
                    claim_extraction.strip() == "No available facts.":
                sentence.claims = []
                continue

            parsed_claim_extraction = utils.parse_claim_extraction(claim_extraction, self.claim_extractor)

            sentence_claims = []
            for claim_text in parsed_claim_extraction:
                if (
                        claim_text.strip() != ""
                        and claim_text not in deduplicate[sentence.generation]
                ):
                    deduplicate[sentence.generation].add(claim_text)
                    claim = Claim(claim=claim_text, \
                                  sentence=sentence, \
                                  refernce=sentence.generation.reference, \
                                  topic=sentence.generation.topic, \
                                  question=sentence.generation.prompt
                                  )
                    sentence_claims.append(claim)
                    all_claims.append(claim)

            sentence.claims = sentence_claims

        for generation in self.generations:
            if not deduplicate[generation]:  # no claims for a generation -> also abstains
                generation.abstain = True

        all_claims_text = [str(c.claim) for c in all_claims]
        utils.save_eval_raw(all_claims_text, output_file=self.parsed_claims_path)

        return all_claims

    def verify_claims(self, all_claims: List[Claim]):
        verification_path = self.verification_path

        claim_verification_res = utils.read_eval_raw(verification_path)

        print("***** [3] Ref Src: ", self.ref_src)
        # 1. Prepare the prompt for verification
        retrieval = LongWikiRetrieval(self.db, cache_base_path=self.CACHE_BASE_PATH,
                                      embed_cache_path=self.embedded_cache_path, \
                                      retrieval_type="gtr-t5-large", batch_size=64)
        questions = list(set([claim.question for claim in all_claims]))
        retrieval.make_ner_cache(questions)
        for claim in tqdm(all_claims):
            passages = retrieval.get_topk_related_passages(topic=claim.topic, claim=claim.claim,
                                                           question=claim.question, k=5)
            context = ""
            for _, psg in enumerate(reversed(passages)):
                context += "Title: {}\nText: {}\n\n".format(psg["title"],
                                                            psg["text"].replace("<s>", "").replace("</s>", ""))
            claim.prompt = self.prompt_templates.VERIFICATION_TEMPLATE_W_REFERENCE_RETRIEVAL.format(claim=claim.claim,
                                                                                                    reference=context)
        print("***** Prepared all verification prompts")

        # 2. Verify the claims
        verification_prompts = [c.prompt for c in all_claims]
        if len(claim_verification_res) == len(all_claims):
            print("***** [3] Reading verification results from cache {}\n".format(verification_path))
        else:
            for i in range(0, len(verification_prompts), 100):
                batch_prompts = verification_prompts[i:i + 100]
                batch_results = utils.model_eval_step(self.verifier, batch_prompts, \
                                                      max_token=512, batch_size=8, \
                                                      max_workers=16)
                claim_verification_res.extend(batch_results)
                utils.save_eval_raw(claim_verification_res, output_file=verification_path)
            utils.save_eval_raw(claim_verification_res, output_file=verification_path)

        assert len(claim_verification_res) == len(all_claims)
        # 3. post process the verification result
        calim_verification_results = utils.jsonify_ans(raw_responses=claim_verification_res, \
                                                       eval_prompts=verification_prompts,
                                                       evaluator=self.verifier, \
                                                       key="is_supported")

        return calim_verification_results


def make_claim_extraction_prompts(generation: Generation, claim_extractor="meta-llama/Llama-3.1-405B-Instruct-FP8",
                                  language='kor'):
    """
    Given a model output
    - split into sentences
    - go para by para, always add the first sent of the para into context1
    - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
    Return list of {"prompt": prompt_text, "sentence": target_sentence}
    """
    if language == 'kor':
        from llm_eval.external.providers.hallulens.tasks.longwiki import ko_prompt_templates as prompt_templates
    elif language == 'eng':
        from llm_eval.external.providers.hallulens.tasks.longwiki import prompt_templates
    else:
        raise ValueError(f"Invalid language: {language}")

    sentences = []
    # split the text into sentences
    sentences_text = [x.strip() for x in split_single(generation.generation)]
    question = generation.prompt.replace("Answer in one paragraph.", "").strip()
    response = generation.generation.strip()

    # TODO: 여기서 뭔가 잘못된거면 애초에 생성이 잘못된거임
    if claim_extractor == "finetuned":
        for i, sentence in list(enumerate(sentences_text)):
            input = f"Questions:\n{question.strip()}\nResponse:\n{response.strip()}"
            snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            prompt_text = prompt_templates.EXTRACT_CLAIMS_TEMPLATE_FINETUNED.format(snippet, "")
            sentences.append(
                Sentence(
                    prompt=prompt_text, sentence=sentence, generation=generation
                )
            )
    elif claim_extractor == "meta-llama/Llama-3.1-405B-Instruct-FP8":
        for i, sentence in list(enumerate(sentences_text)):
            if len(sentence) < 5:
                continue

            target_sentence = sentences_text[i]
            sentence = f"<SOS>{target_sentence.strip()}<EOS>"

            context1 = " ".join(sentences_text[max(0, i - 3): i])
            context2 = " ".join(sentences_text[i + 1: i + 2])
            snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()

            prompt_text = prompt_templates.EXTRACT_CLAIMS_TEMPLATE.format(
                snippet=snippet, sentence=sentence
            )

            # check token 
            prompt_len = len(encoding.encode(prompt_text))
            if prompt_len > 3500:
                context1 = " ".join(sentences_text[max(0, i - 2): i])
                snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()

                prompt_text = prompt_templates.EXTRACT_CLAIMS_SHORT_TEMPLATE.format(
                    snippet=snippet, sentence=sentence
                )

                if len(encoding.encode(prompt_text)) > 3500:

                    prompt_text = prompt_templates.EXTRACT_CLAIMS_EXTREME_SHORT_TEMPLATE.format(
                        snippet=snippet, sentence=sentence
                    )

                    if len(encoding.encode(prompt_text)) > 3500:
                        prompt_text = prompt_templates.EXTRACT_CLAIMS_EXTREME_EXTREME_SHORT_TEMPLATE.format(
                            snippet=snippet, sentence=sentence
                        )

                assert len(encoding.encode(prompt_text)) <= 3500, f"Prompt too long: {len(encoding.encode(prompt_text))}, {prompt_text}"

            sentences.append(
                Sentence(
                    prompt=prompt_text, sentence=target_sentence, generation=generation
                )
            )
    else:
        raise ValueError(f"Invalid claim_extractor: {claim_extractor}")

    generation.sentences = sentences
    return sentences
