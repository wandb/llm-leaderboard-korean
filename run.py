import os

config_list_gpt = [
    #"o1-preview",
    "gpt-3-5-turbo-0125",
    "chatgpt-4o-latest",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0613",
    "gpt-4o-mini-2024-07-18",
    "solar-pro"
]
# config_list_claude = [
#     "anthropic-claude-3-5-sonnet-20240620-v1:0",
#     "anthropic-claude-3-haiku-20240307-v1:0",
#     "anthropic-claude-3-opus-20240229-v1:0",
#     "anthropic-claude-3-sonnet-20240229-v1:0"
# ]
for config in config_list_gpt:
    os.system(f"python3 scripts/run_eval.py --config config-{config}")