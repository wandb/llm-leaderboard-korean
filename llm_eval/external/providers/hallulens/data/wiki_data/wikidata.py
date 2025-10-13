# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import requests
import jsonlines
import pandas as pd
import os
from datasets import load_dataset

################################################################################
# repo_name = "hallucination_benchmark"
# base_path = os.getcwd().split(f'/{repo_name}')[0]
# base_path= ""
################################################################################

class wikiData:
    def __init__(self):
        self.path = ""

        # self.WIKI_DATA_PATH = f"{base_path}/{repo_name}/data/wiki_data"
        self.WIKI_DATA_PATH = f"data/wiki_data"

        self.wikirank_title_path = f"{self.WIKI_DATA_PATH}/enwiki-2024.titles.txt" # https://wikirank-2024.di.unimi.it/rank/enwiki-2024-h.txt
        self.wikirank_hscore_path = f"{self.WIKI_DATA_PATH}/enwiki-2024-h.txt" # https://wikirank-2024.di.unimi.it/enwiki-2024.titles
        self.goodwiki_path = f"{self.WIKI_DATA_PATH}/doc_goodwiki.jsonl"
        self.final_wiki_path = self.goodwiki_path.replace('.jsonl', '_h_score.jsonl')

        self.BINS_RES = [-2.0,
            1339481.3875000002,
            1413480.1812500001,
            1471393.09375,
            1527383.875,
            1582542.25625,
            1636846.01875,
            1689687.5375,
            1750039.75,
            1857754.69375,
            3179443.8875]
        

    def run(self):
        self.download_goodwiki()
        self.download_wikirank()
        self.combine_h_score()

    def download_goodwiki(self):
        print(self.goodwiki_path)
        if os.path.exists(self.goodwiki_path):
            print(f"GoodWiki data already exists at {self.goodwiki_path}")
            return
        
        print("Loading dataset euirim/goodwiki...")
        dataset = load_dataset("euirim/goodwiki")

        print(f"Writing to jsonl...[{self.goodwiki_path}]")
        with jsonlines.open(self.goodwiki_path, "w") as writer:
            for i in range(len(dataset["train"])):
                line = dataset["train"][i]
                line['document'] = line.pop('markdown')
                writer.write(line)
        print("Finished writing to jsonl | Path: ", f"{self.goodwiki_path}")


    def download_wikirank(self):

        if os.path.exists(self.wikirank_hscore_path) and os.path.exists(self.wikirank_title_path):
            print(f"WikiRank data already exists. Pass.")
            return
        
        print(f"Downloading WikiRank data... | Path {self.WIKI_DATA_PATH}")

        # download the data to data/wikirank/enwiki-2024.titles.txt
        wikirank_titles_url = "https://wikirank-2024.di.unimi.it/enwiki-2024.titles"
        r = requests.get(wikirank_titles_url)
        with open(f"{self.WIKI_DATA_PATH}/enwiki-2024.titles.txt", "wb") as f:
            f.write(r.content)
        print("Downloaded enwiki-2024.titles.txt")

        # download the data to data/wikirank/enwiki-2024-h.txt
        wikirank_h_score_url = "https://wikirank-2024.di.unimi.it/rank/enwiki-2024-h.txt"
        r = requests.get(wikirank_h_score_url)
        with open(f"{self.WIKI_DATA_PATH}/enwiki-2024-h.txt", "wb") as f:
            f.write(r.content)
        print("Downloaded enwiki-2024-h.txt")

    def get_wikirank(self):
        with open(self.wikirank_hscore_path, 'r') as f:
            h_scores = [x.strip() for x in f.readlines()]    
        with open(self.wikirank_title_path, 'r') as f:
            titles = [x.strip() for x in f.readlines()]

        assert len(h_scores) == len(titles)

        title_h_scores = dict(zip(titles, h_scores))
        return title_h_scores

    def get_bins(self, df):
        # Define the number of bins
        cut_n = 10

        bins = []
        for i in range(cut_n):
            cut_df = df[int(i*df.shape[0]/10):int((i+1)*df.shape[0]/10)]
            min_h = cut_df['h_score'].min()
            max_h = cut_df['h_score'].max()
            bins.append((min_h, max_h))

        # make them into list of number bins
        bins = [bins[i][0] for i in range(len(bins)-1, -1, -1)] + [bins[0][1]]
        return bins

    def combine_h_score(self):

        bins = self.BINS_RES # obtained by get_bins(df)
        h_scores = self.get_wikirank()
        df = pd.read_json(self.goodwiki_path, lines=True)

        # map the h_score to the goodwiki
        df['h_score'] = df['title'].apply(lambda x: h_scores.get(x, -1))
        df['h_score'] = df['h_score'].astype(float)
        df = df.sort_values(by='h_score', ascending=False)

        # make bins label by code
        bins_label = [f"{x}-{y}" for x, y in zip(bins[:-1], bins[1:])]
        df['h_score_bins'] = pd.cut(df['h_score'], bins=bins, labels=bins_label)
        df['h_score_bins_ratio'] = df['h_score_bins'].map(df['h_score_bins'].value_counts(normalize=True))
        
        # map h_score_bins to easier to read label "h_score_cat"
        df['h_score_cat'] = df['h_score_bins'].map(dict(zip(bins_label, [i for i in range(len(bins_label))])))

        # save doc_goodwiki_h_score.jsonl
        df.to_json(self.goodwiki_path.replace('.jsonl', '_h_score.jsonl'), orient='records', lines=True)
        print("Finished writing to jsonl | Path: ", f"{self.goodwiki_path.replace('.jsonl', '_h_score.jsonl')}")


if __name__ == "__main__":

    wiki = wikiData()
    wiki.run()