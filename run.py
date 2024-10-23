import os

config_list_gpt = [
    # "o1-preview",
    "gpt-3-5-turbo-0125",
    "chatgpt-4o-latest",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0613",
    "gpt-4o-mini-2024-07-18",
    "solar-pro",
    "solar-mini"
]
config_list_claude = [
    "anthropic-claude-3-5-sonnet-20240620-v1:0",
    "anthropic-claude-3-haiku-20240307-v1:0",
    "anthropic-claude-3-opus-20240229-v1:0",
    "anthropic-claude-3-sonnet-20240229-v1:0"
]
config_list_qwen = [
    "Qwen2-5-0-5B-Instruct",
    "Qwen2-5-1-5B-Instruct",
    "Qwen2-5-3B-Instruct",
    "Qwen2-5-7B-Instruct",
    "Qwen2-5-14B-Instruct",
    "Qwen2-5-32B-Instruct",
    "Qwen2-1-5B-Instruct",
    "Qwen2-7B-Instruct",
    # "Qwen2-72B-Instruct",
    # "Qwen1-5-72B-Chat",
]

config_list_korean = [
    "EXAONE-3-0-7-8B-Instruct",
    "EEVE-Korean-Instruct-10-8B-v1-0"
]
for config in ["solar-pro", "solar-mini", "Qwen2-5-14B-Instruct", "Qwen2-5-32B-Instruct"]:
    os.system(f"python3 scripts2/run_eval.py --config config-{config}")