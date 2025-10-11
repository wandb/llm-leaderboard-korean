# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import subprocess
# from datasets import load_dataset
from data.wiki_data.wikidata import wikiData
################################################################################
WIKI_DATA_PATH = f"data/wiki_data"
REFUSAL_PATH = "data/nonexistent_refusal"
################################################################################

def longwiki_eval_data():

    if os.path.exists(f"{WIKI_DATA_PATH}/.cache/enwiki-20230401.db"):
        print("Wikipedia dump already exists. Pass.")
    else:
        command = """
        mkdir -p data/wiki_data/.cache &&
        cd data/wiki_data/.cache &&
        gdown https://drive.google.com/uc?id=1mekls6OGOKLmt7gYtHs0WGf5oTamTNat &&
        cd ../../../
        """
        ret_code = subprocess.run(command, shell=True)
        check_status(ret_code, "data/wiki_data/.cache/enwiki-20230401.db")

    if os.path.exists(f"{WIKI_DATA_PATH}/title_db.jsonl"):
        print("Title db already exists. Pass.")
    else:
        command = """
        echo "making title db..." &&
        python -m data.wiki_data.make_title_db
        """
        ret_code = subprocess.run(command, shell=True)
        check_status(ret_code, "data/wiki_data/.cache")
        
        print("Finished making title db | Path: ", f"{WIKI_DATA_PATH}/title_db.jsonl")
        print("-"*50)


def download_medicine():
    print("Downloading Task3 Medicine data...")
    if os.path.exists(f"{REFUSAL_PATH}/medicine_dataset.csv"):
        print("Medicine data already exists. Pass.")
    else:
        os.makedirs(f"{REFUSAL_PATH}", exist_ok=True)
        subprocess.run([
            "curl", "-L", "-o", f"{REFUSAL_PATH}/medicine_dataset.zip",
            "https://www.kaggle.com/api/v1/datasets/download/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes"
        ])
        subprocess.run(["unzip", f"{REFUSAL_PATH}/medicine_dataset.zip", "-d", f"{REFUSAL_PATH}"])
        os.remove(f"{REFUSAL_PATH}/medicine_dataset.zip")

def download_itis_taxonomy():
    # Step 1: Download the data
    print("Downloading Task4 ITIS data...")

    if os.path.exists(f"{REFUSAL_PATH}/itis_animals.tar.gz"):
        print("ITIS data already exists. Pass.")
    else:
        ret_code = subprocess.run([
            "curl", "-L", "-o", f"{REFUSAL_PATH}/itis_animals.tar.gz",
            "https://www.itis.gov/downloads/itisMySQLTables.tar.gz"
        ])
        check_status(ret_code, f"{REFUSAL_PATH}/itis_animals.tar.gz")
        
    # Step 2: Unzip the itis_animals.tar.gz
    print("Extracting files...")
    os.makedirs(f"{REFUSAL_PATH}/extracted", exist_ok=True)
    os.makedirs(f"{REFUSAL_PATH}/itis_animals", exist_ok=True)
    subprocess.run(["tar", "-xzvf", f"{REFUSAL_PATH}/itis_animals.tar.gz", "-C", f"{REFUSAL_PATH}/extracted"])

    # Step 3: Select files named "longnames" and "hierarchy"
    print("Selecting specific files...")
    for root, dirs, files in os.walk(f"{REFUSAL_PATH}/extracted"):
        for file in files:
            if "longnames" in file or "hierarchy" in file:
                subprocess.run(["cp", os.path.join(root, file), f"{REFUSAL_PATH}/itis_animals"])

    # Step 4: Remove the extracted folder
    print("Cleaning up...")
    subprocess.run(["rm", "-rf", f"{REFUSAL_PATH}/extracted/"])
    
    os.remove(f"{REFUSAL_PATH}/itis_animals.tar.gz")
    print("Process completed.")

    
def check_status(ret_code, path=""):
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(path))
    else:
        print("Download {} ... [Success]".format(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir',
                        type=str,
                        default="./data/")
    
    parser.add_argument("--all",
                        action="store_true",
                        default=False,
                        help="Download all datasets")
    
    parser.add_argument("--precisewikiqa",
                        action="store_true",
                        default=False,
                        help="Prepare Wiki Data for Task 1 and Task 2")
    
    parser.add_argument("--longwiki",
                        action="store_true",
                        default=False,
                        help="Prepare Wikipedia dump for Task 2 Evaluation")
    
    parser.add_argument("--nonexistent_refusal",
                        action="store_true",
                        default=False,
                        help="Download medicine and ITIS data for nonexistent refusal")

    args = parser.parse_args()

    ## Prepare Wiki Data for Task 1 and Task 2
    if args.all or args.precisewikiqa or args.longwiki:
        if not os.path.exists(WIKI_DATA_PATH):
            os.makedirs(WIKI_DATA_PATH)
        wiki_data = wikiData()
        wiki_data.run()
    
    ## Prepare Wikipedia dump for Task 2 Evalution:
    if args.all or args.longwiki:
        longwiki_eval_data()

    # ## Prepare Wikipedia dump for Task 3:
    if args.all or args.nonexistent_refusal:
        if not os.path.exists(REFUSAL_PATH):
            os.makedirs(REFUSAL_PATH)

        ## Download medicine data for nonexistent refusal
        download_medicine()

        ###Download itis_animals for nonexistent refusal
        download_itis_taxonomy()