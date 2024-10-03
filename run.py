import os

config_list = [
    #"o1-preview",
    "chatgpt-4o-latest",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    #"gpt-4-turbo-2024-04-09",
    "gpt-4-0613",
    "gpt-4o-mini-2024-07-18",
    "gpt-3-5-turbo-0125",
]
for config in config_list:
    os.system(f"python3 scripts/run_eval.py --config config-{config}")