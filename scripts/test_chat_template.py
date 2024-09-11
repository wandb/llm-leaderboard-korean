"""
This script generates chat prompts using a specified model and chat template.

Usage:
$ python test_chat_template.py -m <model_id> -c <chat_template>

Examples:

# When the model ID and chat_template are the same
$ python scripts/test_chat_template.py -m "rinna/nekomata-7b-instruction"

# When using a different chat_template than the model ID
$ python scripts/test_chat_template.py -m "rinna/nekomata-7b-instruction" -c "tokyotech-llm/Swallow-MS-v0.1"

Arguments:

- `-m` or `--model-id`: The ID of the model.
- `-c` or `--chat-template`: The ID of the chat template. If omitted, the model ID will be used.
"""

from argparse import ArgumentParser

from jinja2 import Template
from utils import get_tokenizer_config

parser = ArgumentParser()
parser.add_argument("-m", "--model-id", type=str, required=True)
parser.add_argument("-c", "--chat-template", type=str)

model_id = parser.parse_args().model_id
chat_template_name = parser.parse_args().chat_template
if chat_template_name is None:
    chat_template_name = model_id

tokenizer_config = get_tokenizer_config(model_id, chat_template_name)

if chat_template_name.startswith("mistralai/"):
    tokenizer_config.update({"raise_exception": lambda _: ""})
elif chat_template_name.startswith("tokyotech-llm/Swallow") and chat_template_name.endswith("instruct-v0.1"):
    for key in ["bos_token", "eos_token", "unk_token"]:
        if isinstance(tokenizer_config[key], dict):
            tokenizer_config[key] = tokenizer_config[key]["content"]
        else:
            pass

chat_template = Template(tokenizer_config.get("chat_template"))

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant","content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"},
]

prompt = chat_template.render(messages=messages, add_generation_prompt=True, **tokenizer_config)

print(f"""\
------ BEGIN OF PROMPT ------
{prompt}
------- END OF PROMPT -------\
""")
