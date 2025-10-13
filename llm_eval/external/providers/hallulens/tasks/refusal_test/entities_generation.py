# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import pandas as pd
import re
import string
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

DESCRIBE_PROMPTS = ['What can you tell me about the {type} {name}?',
                    'I\'m looking for information on the {type} {name}.',
                    'Can you describe the {type} {name}?',
                    'I want to know more about the {type} {name}.',
                    'Could you give me details about the {type} {name}?',
                    'What do you know about the {type} {name}?',
                    "Can you share some details about the {type} {name}?",
                    "I'm curious about the {type} {name}. What can you tell me?",
                    "I'd like to learn more about the {type} {name}.",
                    "Could you provide an overview of the {type} {name}?"]

KO_DESCRIBE_PROMPTS = ['{type} {name}에 대해 알려주시겠어요?',
                       '{type} {name}에 대한 정보를 찾고 있어요.',
                       '{type} {name}에 대해 설명해주세요.',
                       '{type} {name}에 대해 더 알고 싶어요.',
                       '{type} {name}에 대해 자세히 알려주시겠어요?',
                       '{type} {name}에 대해 아는 것을 알려주세요.',
                       '{type} {name}에 대한 세부 정보를 공유해주실 수 있나요?',
                       '{type} {name}이 궁금해요. 무엇을 알려줄 수 있나요?',
                       '{type} {name}에 대해 더 배우고 싶어요.',
                       '{type} {name}에 대한 개요를 설명해주시겠어요?'
                       ]


class NonsenseMixedGeneration:
    def __init__(self, seed: int, N: int = 2000, EXP: str = "all", language='kor'):
        self.seed = seed
        self.N = N
        self.EXP = EXP
        print('will generate N per domain:', self.N)

        self.generators = {
            'medicine': NonsenseMedicineGeneration(self.seed, self.N),
            'animal': NonsenseTaxonomyGeneration(self.seed, self.N, kingdom_type='animal'),
            'plant': NonsenseTaxonomyGeneration(self.seed, self.N, kingdom_type='plant'),
            'bacteria': NonsenseTaxonomyGeneration(self.seed, self.N, kingdom_type='bacteria')
        }

        if language == 'kor':
            self.describe_prompt = KO_DESCRIBE_PROMPTS
        elif language == 'en':
            self.describe_prompt = DESCRIBE_PROMPTS
        else:
            raise ValueError(f"Unsupported language: {language}")

    def generate_prompts(self):
        types = {
            "nonsense_all": ['medicine', 'animal', 'plant', 'bacteria'],
            "nonsense_medicine": ['medicine'],
            "nonsense_animal": ['animal'],
            "nonsense_plant": ['plant'],
            "nonsense_bacteria": ['bacteria']
        }.get(self.EXP, [])

        if not types:
            raise ValueError(f"Invalid EXP: {self.EXP}")

        prompt_objs = []
        for type_ in types:
            generator = self.generators[type_]
            nonsense_names = generator.makeup_names()
            prompt_objs += self.make_prompt_objs(nonsense_names, type_)
        return prompt_objs

    def make_prompt_objs(self, names: list[str], type_: str):
        prompt_objs = [
            {"prompt": random.choice(self.describe_prompt).format(type=type_, name=name),
             "name": name,
             "type": type_
             } for name in names
        ]
        return prompt_objs

    def save_prompt_csv(self,
                        prompt_objs: list[dict],
                        prompt_path: str):

        prompt_dir = '/'.join(prompt_path.split('/')[:-1])
        os.makedirs(prompt_dir, exist_ok=True)

        df = pd.DataFrame(prompt_objs)
        df.to_csv(prompt_path, index=False)


# ============= ANIMAL /PLANT / BACTERIA =============

class Node:
    def __init__(self, code, name=None, parent=None):
        self.name = name
        self.code = code
        self.parent = parent
        self.children = []
        self.is_family = False

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return f"{self.name}"


class NonsenseTaxonomyGeneration:
    def __init__(self, seed, N, kingdom_type):
        # self.SPECIES, self.ALL_SPECIES_NAME = self.make_graph()
        self.seed = seed
        self.N = N
        self.type = kingdom_type
        self.DATAPATH = "data/nonexistent_refusal/itis_animals"

        if 'animal' == kingdom_type:
            self.kingdom = 'Animalia'
        elif 'plant' == kingdom_type:
            self.kingdom = 'Plantae'
        elif 'bacteria' == kingdom_type:
            self.kingdom = 'Bacteria'
        else:
            raise ValueError(f'Invalid kingdom type: {kingdom_type}')

    def make_graph(self):
        graph = Node("root")
        node_lookup = {}
        with Path(f"{self.DATAPATH}/hierarchy").open() as f:
            for line in tqdm(f):
                parent = graph
                for code in line.split("|")[0].strip().split("-"):
                    code = int(code)
                    if code in node_lookup:
                        node = node_lookup[code]
                    else:
                        node = Node(code)
                        node.parent = parent
                        parent.add_child(node)
                        node_lookup[code] = node
                    parent = node

        all_species_names = set()
        with Path(f"{self.DATAPATH}/longnames").open(encoding="ISO-8859-1") as f:
            try:
                for line in tqdm(f):
                    code, name = line.split("|")
                    code = int(code)
                    if code in node_lookup:
                        node = node_lookup[code]
                        node.name = name.strip()
                        all_species_names.add(name.strip())
            except:
                print(line)
                raise

        species = [v for v in node_lookup.values() if len(v.children) == 0]
        return species, all_species_names

    def get_kingdom(self, node):
        while (node.parent and node.parent.parent):
            node = node.parent
        return node

    def makeup_names(self):
        self.SPECIES, self.ALL_SPECIES_NAME = self.make_graph()
        kingdom = self.kingdom
        random.seed(self.seed)
        species_sample = set()
        while len(species_sample) < self.N:
            while True:
                sample = random.choice(self.SPECIES)
                # print(get_kingdom(sample).name)
                if len(sample.name.split(" ")) == 2 and self.get_kingdom(sample).name == kingdom:
                    break
            other_genuses = [uncle for uncle in sample.parent.parent.children if
                             uncle != sample.parent and len(uncle.children) > 0]
            if len(other_genuses) == 0:
                continue
            other_genus = random.choice(other_genuses)
            made_up_name = other_genus.name + " " + sample.name.split(" ")[1]
            # print("Combining", sample.name, "with", other_genus.name, "to make", made_up_name)
            if made_up_name in self.ALL_SPECIES_NAME:
                continue
            else:
                # return made_up_name
                species_sample.add(made_up_name)
        return species_sample


# ======= MEDICINE ===========
class NonsenseMedicineGeneration:
    def __init__(self, seed, N):
        self.seed = seed
        # N total number of names to generate
        self.N = N
        self.DATAPATH = "data/nonexistent_refusal/medicine_dataset.csv"
        print('will generate N:', self.N)
        self.get_orinal_names()
        self.get_word_pool()

    def get_orinal_names(self):
        df = pd.read_csv(self.DATAPATH)
        name = df['name']
        existing_names = [str(x).lower() for x in name]
        substitute0 = df['substitute0']
        existing_names += [str(x).lower() for x in substitute0]
        substitute1 = df['substitute1']
        existing_names += [str(x).lower() for x in substitute1]
        substitute2 = df['substitute2']
        existing_names += [str(x).lower() for x in substitute2]
        substitute3 = df['substitute3']
        existing_names += [str(x).lower() for x in substitute3]
        substitute4 = df['substitute4']
        existing_names += [str(x).lower() for x in substitute4]

        existing_names2 = []
        for name in existing_names:
            # remove the part in ()
            name = re.sub(r'\(.*?\)', '', name)
            if "," in name or '&' in name:
                continue

            words = name.split()
            words2 = []
            for w in words:
                # skip w if number in w
                if any(char.isdigit() for char in w):
                    continue
                words2.append(w)
            name = ' '.join(words2)
            existing_names2.append(name)
        self.existing_names = existing_names2

    def get_word_pool(self):
        first_word_pool = defaultdict(set)
        second_word_pool = defaultdict(set)
        third_word_pool = defaultdict(set)
        for name in self.existing_names:
            parts = name.split()
            length = len(parts)
            first_word_pool[length].add(parts[0])
            if length > 1:
                second_word_pool[length].add(parts[1])
            if length > 2:
                third_word_pool[length].add(parts[2])

        # turn set to list
        for length in first_word_pool:
            first_word_pool[length] = list(first_word_pool[length])
        for length in second_word_pool:
            second_word_pool[length] = list(second_word_pool[length])
        for length in third_word_pool:
            third_word_pool[length] = list(third_word_pool[length])
        self.first_word_pool, self.second_word_pool, self.third_word_pool = first_word_pool, second_word_pool, third_word_pool

    def generate_similar_name(self, original_name):
        parts = original_name.split()
        length = len(parts)
        if length > 3:
            return ""
        new_parts = []
        new_parts.append(random.choice(self.first_word_pool[length]))
        if length > 1:
            new_parts.append(random.choice(self.second_word_pool[length]))
        if length > 2:
            new_parts.append(random.choice(self.third_word_pool[length]))
        return ' '.join(new_parts)

    def makeup_names(self):
        existing_names = self.existing_names
        random.seed(self.seed)
        new_names = set()
        with tqdm(total=self.N, desc="Generating Names") as pbar:
            while len(new_names) < self.N:
                original_name = random.choice(existing_names)
                new_name = self.generate_similar_name(original_name)
                if new_name and new_name not in existing_names and new_name not in new_names:
                    new_names.add(new_name)
                    pbar.update(1)

        new_names = list(new_names)
        return new_names
