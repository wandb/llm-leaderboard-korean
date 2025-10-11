# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import json
import re
import random
import math
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from llm_eval.external.providers.hallulens.utils import exp

from llm_eval.external.providers.hallulens.tasks.refusal_test.nonsense_name import NonsenseNameInference, NonsenseNameEval
from llm_eval.external.providers.hallulens.tasks.refusal_test.search import batch_search

# 30 * 3 level  google num results
CITIES = {'high': ['New York', 'London', 'Washington', 'Chicago', 'Paris', 'Columbia', 'Boston', 'Philadelphia',
                   'Los Angeles', 'Rome', 'San Francisco', 'Moscow', 'Berlin', 'Toronto', 'Police', 'Douglas',
                   'Detroit', 'Baltimore', 'Delhi', 'Marshall', 'Tokyo', 'Houston', 'Singapore', 'Hong Kong', 'Atlanta',
                   'Sydney', 'Denver', 'Seattle', 'Dallas', 'Vienna'],
          'mid': ['Ikeda', 'Guadalajara', 'Jakarta', 'Rosario', 'Emporia', 'Despatch', 'Tianjin', 'Ennis', 'Sakai',
                  'Jeddah', 'Rangoon', 'Dar es Salaam', 'Addis Ababa', 'Choctaw', 'Ahmedabad', 'Quezon City', 'Lucknow',
                  'Khartoum', 'Chennai', 'Pune', 'Milford', 'São Paulo', 'Grimes', 'Sioux City', 'Bainbridge',
                  'Merrick', 'Middletown', 'Abidjan', 'Guatemala City', 'St. Peter'],
          'low': ['Friesenheim', 'Portet-sur-Garonne', 'Fernán-Núñez', 'Matías Romero', 'Steenokkerzeel',
                  'Atotonilco el Alto', 'Lenyenye', 'Ambohinamboarina', 'Borim', 'Antaly', 'Eséka', 'Velaux', 'Māndu',
                  'Brugnera', 'Pezinok', 'Janglot', 'Morab', 'Yepocapa', 'Quảng Trị', 'Nanchong', 'As Sars',
                  'Santa Teresa di Riva', 'Arganil', 'Maitum', 'Xinpu', 'Marquetalia', 'Dazhou', 'Ambatomena', 'Painan',
                  'Travilah']}

COUNTRIES = {
    'high': ['United States', 'India', 'Canada', 'China', 'France', 'Japan', 'Germany', 'Mexico', 'Australia', 'Israel',
             'Italy', 'Russia', 'Ireland', 'Spain', 'United Kingdom', 'Georgia', 'Pakistan', 'Egypt', 'South Africa',
             'Vietnam', 'Netherlands', 'New Zealand', 'Brazil', 'Sweden', 'Poland', 'Switzerland', 'Greece', 'Turkey',
             'Philippines', 'Singapore'],
    'mid': ['Tanzania', 'Iceland', 'South Korea', 'Ecuador', 'Bolivia', 'Uruguay', 'Costa Rica', 'Libya', 'Haiti',
            'Sri Lanka', 'Zimbabwe', 'Angola', 'Zambia', 'Laos', 'Honduras', 'El Salvador', 'Tunisia', 'Malta', 'Fiji',
            'Liberia', 'Mozambique', 'Samoa', 'Senegal', 'Lithuania', 'Namibia', 'Sierra Leone', 'Barbados', 'Yemen',
            'Serbia', 'Chad'],
    'low': ['Eritrea', 'Micronesia', 'Seychelles', 'Solomon Islands', 'United Arab Emirates', 'Dominica', 'Palau',
            'Moldova', 'Holy See', 'Tajikistan', 'Liechtenstein', 'Bosnia and Herzegovina', 'Suriname',
            'Marshall Islands', 'Turkmenistan', 'Vanuatu', 'Central African Republic', 'Djibouti', 'Burkina Faso',
            'San Marino', 'Nauru', 'Kyrgyzstan', 'Maldives', 'Equatorial Guinea', 'Guinea-Bissau', 'Andorra',
            'Kiribati', 'Comoros', 'Democratic Republic of the Congo', 'Tuvalu']}

BUSINESS_TYPES = ["restaurant", 'bar', 'bookstore', 'cafe', 'museum']  # 5 
PRODUCT_TYPES = ["headphone", 'watch', 'camera', 'shoe', "game", "glasses", 'speaker', 'keyboard', 'chair', 'table',
                 'printer', 'pen', 'guitar', 'piano', 'bike', 'drone', 'coffee maker', 'projector', 'bag',
                 'air conditioner',
                 'bread', 'microwave', 'vacuum cleaner', 'sewing machine', 'router']  # 25
HISTORICAL_EVENT_TYPES = ["war", "natural disaster", "scientific discovery", "sport event", "pandemic", ]  # 5

MODELS = ["Llama-3.1-405B-Instruct", "Mistral-Nemo-Instruct-2407", "gpt-4o"]
FULL_NAME_MODELS = ["meta-llama/Llama-3.1-405B-Instruct-FP8", "mistralai/Mistral-Nemo-Instruct-2407", "gpt-4o"]

DOMAIN_DICT = {}


def call_domain_dict(prompt_templates):
    return {
        "business": (CITIES, BUSINESS_TYPES, prompt_templates.BUSINESS_PROMPT),
        "event": (COUNTRIES, HISTORICAL_EVENT_TYPES, prompt_templates.HISTORICAL_EVENT_PROMPT),
        "product": ([""], PRODUCT_TYPES, prompt_templates.PRODUCT_PROMPT),
    }


class NonsenseNameGeneration:
    def __init__(self, seed,
                 BUSINESS_N, EVENT_N, PRODUCT_N,
                 BUSINESS_NAME_NUM, EVENT_NAME_NUM, PRODUCT_NAME_NUM,
                 round_robin_pattern, root_path, method,
                 language='kor'):
        self.seed = seed
        # N total number of names to generate
        self.BUSINESS_N = BUSINESS_N
        self.EVENT_N = EVENT_N
        self.PRODUCT_N = PRODUCT_N
        self.NS = {"business": self.BUSINESS_N, "event": self.EVENT_N, "product": self.PRODUCT_N}
        print('will generate BUSINESS_N:', self.BUSINESS_N, 'EVENT_N:', self.EVENT_N, 'PRODUCT_N:', self.PRODUCT_N)

        # number of names to generate per prompt
        self.BUSINESS_NAME_NUM = BUSINESS_NAME_NUM
        self.EVENT_NAME_NUM = EVENT_NAME_NUM
        self.PRODUCT_NAME_NUM = PRODUCT_NAME_NUM
        self.NAME_NUMS = {"business": self.BUSINESS_NAME_NUM, "event": self.EVENT_NAME_NUM,
                          "product": self.PRODUCT_NAME_NUM}
        assert round_robin_pattern in ['average', 'mixtral']
        self.round_robin_pattern = round_robin_pattern
        self.root_path = root_path
        self.inference_method = method

        # Adjust models because TogetherAI doesn't support Mistral-Nemo-Instruct-2407
        if self.inference_method == 'together':
            MODELS[MODELS.index('Mistral-Nemo-Instruct-2407')] = "Mistral-Small-24B-Instruct-2501"
            FULL_NAME_MODELS[FULL_NAME_MODELS.index('mistralai/Mistral-Nemo-Instruct-2407')] = "mistralai/Mistral-Small-24B-Instruct-2501"

        # Prompt templates according to language
        if language == 'kor':
            import tasks.refusal_test.ko_prompt as prompt_templates
        elif language == 'eng':
            import tasks.refusal_test.prompt as prompt_templates
        else:
            raise NotImplementedError
        self.prompt_templates = prompt_templates

        global DOMAIN_DICT
        DOMAIN_DICT = call_domain_dict(prompt_templates)

    def sample_places_types(self):
        ####### sample places and types #######
        self.SELECTED_PLACES = {}
        self.SELECTED_TYPES = {}
        for domain in DOMAIN_DICT.keys():
            places = self.sample_places(domain)
            types = self.sample_types(domain)
            self.SELECTED_PLACES[domain] = places
            self.SELECTED_TYPES[domain] = types
            print('domain', domain, 'places', len(places), 'types', len(types), self.NAME_NUMS[domain])

    def sample_places(self, domain):
        all_places, types, _ = DOMAIN_DICT[domain]
        dN, dNAME_NUM = self.NS[domain], self.NAME_NUMS[domain]

        if len(all_places) <= 1:
            assert type(all_places) == list
            return all_places

        num_place = math.ceil(dN / dNAME_NUM / len(types))
        if num_place * dNAME_NUM * len(types) * 0.95 < dN:
            self.NAME_NUMS[domain] = math.ceil(
                dN / 0.95 / num_place / len(types))  # sample more to make sure enough after web check

        # three level
        random.seed(self.seed)
        high_places = random.sample(all_places['high'], num_place // 3)
        mid_places = random.sample(all_places['mid'], num_place // 3)
        low_places = random.sample(all_places['low'], num_place // 3)
        places = random.sample(all_places['high'] + all_places['mid'] + all_places['low'], num_place % 3)
        places += high_places + mid_places + low_places
        print('sampled', len(places), 'places')
        return sorted(places)

    def sample_types(self, domain):
        _, types, _ = DOMAIN_DICT[domain]
        if domain != "product":
            return sorted(types)

        dN, dNAME_NUM = self.NS[domain], self.NAME_NUMS[domain]
        type_num = math.ceil(dN / dNAME_NUM / 1)
        if type_num * dNAME_NUM * 0.95 < dN:
            self.NAME_NUMS[domain] = math.ceil(
                dN / 0.95 / type_num / 1)  # sample more to make sure enough after web check
        random.seed(self.seed)
        types = random.sample(types, type_num)
        print('sampled', len(types), 'types')
        return sorted(types)

    def get_inital_model_prompts(self, domain):
        places, types = self.SELECTED_PLACES[domain], self.SELECTED_TYPES[domain]
        NUM, PROMPT = self.NAME_NUMS[domain], DOMAIN_DICT[domain][2]

        all_prompts = []
        for type_ in types:
            for place in places:
                prompt = PROMPT.format(PLACE=place, TYPE=type_, NUM=NUM)
                all_prompts.append(prompt)
        return all_prompts

    def get_final_model_prompts(self, domain, all_not_exist1, all_not_exist2):
        places = self.SELECTED_PLACES[domain]
        types = self.SELECTED_TYPES[domain]
        NUM = self.NAME_NUMS[domain]

        old_not_exist = dict()
        for type_ in types:
            for place in places:
                names1 = all_not_exist1[type_][place]
                names2 = all_not_exist2[type_][place]
                old_not_exist[f"{type_}_{place}"] = names1 + names2

        all_prompts = []
        for type_ in types:
            for place in places:
                place2 = " in " + place if place else ""
                names = old_not_exist[f"{type_}_{place}"]
                names = ', '.join(names)
                if domain == "product":
                    prompt = self.prompt_templates.MIX_PROMPT_PRDOUCT
                else:
                    prompt = self.prompt_templates.MIX_PROMPT
                prompt = prompt.format(type_=type_, place=place2, names=names, NUM=NUM)
                all_prompts.append(prompt)
        return all_prompts

    def double_check_with_web(self, domain, all_repies, skip=False):
        places = self.SELECTED_PLACES[domain]
        types = self.SELECTED_TYPES[domain]

        all_not_exist = defaultdict(list)  # all_not_exist[TYPE_PLACE] = [name1, name2, ...]
        i = 0
        for type_ in tqdm(types):
            for place in places:
                reply = all_repies[i]
                i += 1
                # if f"{type_}_{place}" in all_not_exist:
                #     continue # history
                names = self.process_name_list(reply, self.NAME_NUMS[domain])
                if skip:
                    all_not_exist[f"{type_}_{place}"] = names
                    continue

                queries = []
                for name in names:
                    query = f"{type_} named {name}"
                    if place:
                        query += f" in {place}"
                    queries.append(query)

                all_search_results = batch_search(queries,
                                                  max_workers=16)
                time.sleep(0.5) # to avoid rate limit

                assert len(all_search_results) == len(names)
                for name, search_results in zip(names, all_search_results):
                    exist = False
                    for r in search_results["search_result"][:3]:
                        title = r['title'].strip()
                        snippet = r['snippet'].strip()
                        if name in title and type_ in snippet.lower():
                            # print(title)
                            # print(r['link'])
                            exist = True
                            break
                    if not exist:
                        all_not_exist[f"{type_}_{place}"].append(name)

        # sort by k
        all_not_exist = dict(sorted(all_not_exist.items(), key=lambda x: x[0]))
        all_not_exist2 = defaultdict(dict)
        for k, v in all_not_exist.items():
            type_, place = k.split('_')
            all_not_exist2[type_][place] = v

        return all_not_exist2

    def process_name_list(self, reply, NUM):
        reply = str(reply)
        reply2 = re.split(r":\W+", reply)[-1].strip()
        reply2 = re.split(r"\n\n+", reply2)[0]
        names = reply2.split(',')
        names = [name.strip() for name in names]

        if len(names) != NUM:
            names = reply2.split('\n')

        if len(names) != NUM:
            names = reply2.split('," "')
            names[0] = names[0][1:]
            names[-1] = names[-1][:-2]

        if len(names) != NUM:
            # print("error", reply)
            print(len(names), NUM)
        # assert len(names) == NUM

        names2 = []
        for name in set(names):
            name = re.sub(r"\d+\. ", "", name)
            name = name.strip()
            names2.append(name)

        return names2

    def save_json_file(self, data, file_path):
        out_folder = '/'.join(file_path.split('/')[:-1])
        os.makedirs(out_folder, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f)

    def generate_name_per_model(self, input_file, model_index, all_prompts, domain, skip_check):
        if os.path.exists(input_file):
            with open(input_file) as f:
                all_not_exist = json.load(f)
        else:
            if os.path.exists(input_file[:-5] + "_before_num_limit.json"):
                all_not_exist = json.load(open(input_file[:-5] + "_before_num_limit.json"))
            else:
                all_repies_cached_path = input_file[:-5] + "_all_repies.jsonl"
                all_repies = []  # load history of all repies
                if os.path.exists(all_repies_cached_path):
                    all_repies = pd.read_json(all_repies_cached_path, lines=True)['generation'].to_list()
                if len(all_repies) < len(all_prompts):  # generate the rest
                    all_prompts2 = pd.DataFrame([{"prompt": p} for p in all_prompts])
                    if FULL_NAME_MODELS[model_index] == "gpt-4o":
                        inference_method = "openai"
                    else:
                        inference_method = self.inference_method
                    all_repies = exp.run_exp("TASKNAME",
                                             FULL_NAME_MODELS[model_index],
                                             all_prompts2,
                                             generations_file_path=all_repies_cached_path,
                                             inference_method=inference_method,
                                             max_tokens=1000,
                                             return_gen=True)

                    all_repies = all_repies['generation'].to_list()
                assert len(all_repies) == len(all_prompts)
                all_not_exist = self.double_check_with_web(domain, all_repies, skip=skip_check)
                self.save_json_file(all_not_exist, input_file[:-5] + "_before_num_limit.json")

            all_not_exist = self.limit_sample_num(all_not_exist, self.NS[domain])
            self.save_json_file(all_not_exist, input_file)
        return all_not_exist

    def round_robin_name_generate_per_sequence(self, mi):
        # for one sequence of entity generators
        # mi is the shuffle model index
        all_domain_not_exist_names = dict()  # [type][place] = name list
        for domain in DOMAIN_DICT.keys():
            root_path = self.root_path + f"/{domain}_{self.seed}_{self.NS[domain]}"
            os.makedirs(root_path, exist_ok=True)
            print("domain", domain)

            all_prompts = self.get_inital_model_prompts(domain)

            all_not_exists = []
            for idx, m_idx in enumerate([mi - 2, mi - 1]):
                print(f'{idx}: Entity name generation by', MODELS[m_idx])
                input_file = f"{root_path}/{MODELS[m_idx]}_all_not_exist.json"
                all_not_exist_x = self.generate_name_per_model(input_file, m_idx, all_prompts, domain, skip_check=True)
                all_not_exists.append(all_not_exist_x)
            all_not_exist1, all_not_exist2 = all_not_exists[0], all_not_exists[1]

            print('Final model name generation by', mi, MODELS[mi])
            all_prompts = self.get_final_model_prompts(domain, all_not_exist1, all_not_exist2)
            final_name_file = f"{root_path}/{MODELS[mi]}_{MODELS[mi - 2]}_{MODELS[mi - 1]}_all_not_exist.json"
            all_not_exist = self.generate_name_per_model(final_name_file, mi, all_prompts, domain, skip_check=False)
            all_domain_not_exist_names.update(all_not_exist)

        ## merge all domain
        output_name_file = f"{self.root_path}/merged/{self.seed}_{self.BUSINESS_N}_{self.EVENT_N}_{self.PRODUCT_N}_{MODELS[mi]}_{MODELS[mi - 2]}_{MODELS[mi - 1]}_all_not_exist.json"
        self.save_json_file(all_domain_not_exist_names, output_name_file)
        return output_name_file

    def limit_sample_num(self, all_not_exist, N):
        # total number of names is Norinal_num = 0
        orinal_num = 0
        for type_, places_names in all_not_exist.items():
            for place, names in places_names.items():
                orinal_num += len(names)
        print('orinal_num', orinal_num, 'N', N)

        if orinal_num < N:
            print('not enough names!!!!', orinal_num, N)
            return all_not_exist

        remove_num = orinal_num - N
        while remove_num > 0:
            for type_, places_names in all_not_exist.items():
                sorted_places_names = sorted(places_names.items(), key=lambda x: len(x[1]), reverse=True)
                for place, names in sorted_places_names:
                    if remove_num <= 0:
                        break
                    if len(names) > 1:
                        names.pop()  # remove the last one
                        remove_num -= 1

        return all_not_exist

    def round_robin_name_generate(self, prompt_path):
        self.sample_places_types()
        generated_entities_files = []
        assert self.round_robin_pattern == 'average'
        for mi, m in enumerate(MODELS):
            output_name_file = self.round_robin_name_generate_per_sequence(mi)
            generated_entities_files.append(output_name_file)
            # f"{self.root_path}/merged/{self.seed}_{self.BUSINESS_N}_{self.EVENT_N}_{self.PRODUCT_N}_{MODELS[mi]}_{MODELS[mi-2]}_{MODELS[mi-1]}_all_not_exist.json"
        prompt_path2 = self.save_prompt_file(generated_entities_files)
        assert prompt_path == prompt_path2

    def save_prompt_file(self, generated_entities_files):
        all_prompts = []
        for generated_entities_file in generated_entities_files:
            with open(generated_entities_file) as f:
                all_not_exist = json.load(f)
            shuffle_model = generated_entities_file.split('/')[-1].split('_')[4]
            all_prompts_NUM = sum(
                [len(names) for type_, places_names in all_not_exist.items() for place, names in places_names.items()])
            random.seed(self.seed)
            prompt_idxes = [random.randint(0, len(self.prompt_templates.DESCRIBE_PROMPTS) - 1) for _ in
                            range(all_prompts_NUM)]

            current_num = 0
            for type_, places_names in all_not_exist.items():
                for place, names in places_names.items():
                    for name in names:
                        prompt_idx = prompt_idxes[current_num]
                        place2 = " in " + place if place else ""
                        if type_ in PRODUCT_TYPES:
                            PROMPT = self.prompt_templates.DESCRIBE_PROMPTS_PRODUCT[prompt_idx]
                        else:
                            PROMPT = self.prompt_templates.DESCRIBE_PROMPTS[prompt_idx]

                        all_prompts.append({
                            "place": place,
                            "type_": type_,
                            "name": name,
                            "prompt": PROMPT.format(type_=type_, place=place2, name=name),
                            "seed": self.seed,
                            "shuffle_model": shuffle_model, })

        os.makedirs(f"{self.root_path}/save", exist_ok=True)
        prompt_path = f"{self.root_path}/save/{self.seed}_{self.BUSINESS_N}_{self.EVENT_N}_{self.PRODUCT_N}_all_not_exist.csv"
        all_prompts = pd.DataFrame(all_prompts)
        all_prompts.to_csv(prompt_path, index=False)
        print('saved prompts into: ', prompt_path)
        return prompt_path

    def remove_existing_files(self):
        # remove before running.
        # during running, we can still re-use the intermediate files
        for mi, m in enumerate(MODELS):
            all_rm_files = []
            for domain in DOMAIN_DICT.keys():
                root_path = self.root_path + f"/{domain}_{self.seed}_{self.NS[domain]}"
                input_file1 = f"{root_path}/{MODELS[mi - 2]}_all_not_exist.json"
                input_file2 = f"{root_path}/{MODELS[mi - 1]}_all_not_exist.json"
                final_name_file = f"{root_path}/{MODELS[mi]}_{MODELS[mi - 2]}_{MODELS[mi - 1]}_all_not_exist.json"
                all_rm_files.extend([input_file1, input_file2, final_name_file])
            output_name_file = f"{self.root_path}/{self.seed}_{self.BUSINESS_N}_{self.EVENT_N}_{self.PRODUCT_N}_{MODELS[mi]}_{MODELS[mi - 2]}_{MODELS[mi - 1]}_all_not_exist.json"
            all_rm_files.append(output_name_file)

            for input_file in all_rm_files:
                remove_file(input_file)
                remove_file(input_file[:-5] + "_before_num_limit.json")
                remove_file(input_file[:-5] + "_all_repies.jsonl")
        print('All historical files removed')


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
