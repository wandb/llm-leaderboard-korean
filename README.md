# Horangi Leaderboard 3
## Overview

This repository is for the Horangi Leaderboard 3, a comprehensive evaluation platform for large language models. The leaderboard assesses both general language capabilities and alignment aspects. For detailed information about the leaderboard, please visit [Horangi: W&B Korean LLM Leaderboard 3](horangi.ai) website.

## Implementation Guide

### Environment Setup
1. Set up environment variables
```
export WANDB_API_KEY=<your WANDB_API_KEY>
export MKL_THREADING_LAYER=GNU
export LANG=ko_KR.UTF-8
# if needed, set the following API KEY too
export OPENAI_API_KEY=<your OPENAI_API_KEY>
export ANTHROPIC_API_KEY=<your ANTHROPIC_API_KEY>
export GOOGLE_API_KEY=<your GOOGLE_API_KEY>
export COHERE_API_KEY=<your COHERE_API_KEY>
export MISTRAL_API_KEY=<your MISTRAL_API_KEY>
export AWS_ACCESS_KEY_ID=<your AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<your AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<your AWS_DEFAULT_REGION>
export UPSTAGE_API_KEY=<your UPSTAGE_API_KEY>
# if needed, please login in huggingface
huggingface-cli login
```

2. Clone the repository
```bash
git clone https://github.com/wandb/llm-leaderboard-korean
cd llm-leaderboard
```

3. Set up a Python environment
```bash
sudo apt install -y build-essential
pip install -r requirements.txt
```


### Configuration

#### Base configuration

The `base_config.yaml` file contains basic settings, and you can create a separate YAML file for model-specific settings. This allows for easy customization of settings for each model while maintaining a consistent base configuration.

Below, you will find a detailed description of the variables utilized in the `base_config.yaml` file.

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `entity`: Name of the W&B Entity.
    - `project`: Name of the W&B Project.
    - `run_name`: Name of the W&B run. Please set up run name in a model-specific config.
- **testmode:** Default is false. Set to true for lightweight implementation with a small number of questions per category (for functionality checks).
- **inference_interval:** Set inference interval in seconds. This is particularly effective when there are rate limits, such as with APIs.
- **run:** Set to true for each evaluation dataset you want to run.
- **model:** Information about the model.
    - `artifacts_path`: Path of the wandb artifacts where the model is located.
    - `max_model_len`: Maximum token length of the input.
    - `chat_template`: Path to the chat template file. This is required for open-weights models.
    - `dtype`: Data type. Choose from float32, float16, bfloat16.
    - `trust_remote_code`:  Default is true.
    - `device_map`: Device map. Default is "auto".
    - `load_in_8bit`: 8-bit quantization. Default is false.
    - `load_in_4bit`: 4-bit quantization. Default is false.

- **generator:** Settings for generation. For more details, refer to the [generation_utils](https://huggingface.co/docs/transformers/internal/generation_utils) in Hugging Face Transformers.
    - `top_p`: top-p sampling. Default is 1.0.
    - `temperature`: The temperature for sampling. Default is 0.1.
    - `max_tokens`: Maximum number of tokens to generate. This value will be overwritten in the script.

- **num_few_shots:**  Number of few-shot examples to use.

- **github_version:** For recording, not required to be changed.

- **kaster:**  Settings for the Kaster dataset.
    - `artifacts_path`: URL of the WandB Artifact for the Kaster dataset.
    - `dataset_dir`: Directory of the Kaster dataset after downloading the Artifact.

- **haerae_bench_v1:**  Settings for the haerae_bench_v1 dataset.
    - `artifacts_path`: URL of the WandB Artifact for the haerae_bench_v1 dataset.
    - `dataset_dir`: Directory of the haerae_bench_v1 dataset after downloading the Artifact.

- **kobbq:** Settings for the KoBBQ dataset.
    - `artifacts_path`: URL of the WandB Artifact for the KoBBQ dataset.
    - `dataset_dir`: Directory of the KoBBQ dataset after downloading the Artifact.

- **ko_truthful_qa:** Settings for the KoTruthfulQA dataset.
    - `artifacts_path`: URL of the WandB Artifact for the KoTruthfulQA dataset.
    - `dataset_dir`: Directory of the KoTruthfulQA dataset after downloading the Artifact.

- **mtbench:** Settings for the MT-Bench evaluation.
    - `temperature_override`: Override the temperature for each category of the MT-Bench.
    - `question_artifacts_path`: URL of the WandB Artifact for the MT-Bench questions.
    - `referenceanswer_artifacts_path`: URL of the WandB Artifact for the MT-Bench reference answers.
    - `judge_prompt_artifacts_path`: URL of the WandB Artifact for the MT-Bench judge prompts.
    - `bench_name`: Choose 'japanese_mt_bench' for the Japanese MT-Bench, or 'mt_bench' for the English version.
    - `model_id`: The name of the model. You can replace this with a different value if needed.
    - `question_begin`: Starting position for the question in the generated text.
    - `question_end`: Ending position for the question in the generated text.
    - `max_new_token`: Maximum number of new tokens to generate.
    - `num_choices`: Number of choices to generate.
    - `num_gpus_per_model`: Number of GPUs to use per model.
    - `num_gpus_total`: Total number of GPUs to use.
    - `max_gpu_memory`: Maximum GPU memory to use (leave as null to use the default).
    - `dtype`: Data type. Choose from None, float32, float16, bfloat16.
    - `judge_model`: Model used for judging the generated responses. Default is `gpt-4o-2024-05-13`
    - `mode`: Mode of evaluation. Default is 'single'.
    - `baseline_model`: Model used for comparison. Leave as null for default behavior.
    - `parallel`: Number of parallel threads to use.
    - `first_n`: Number of generated responses to use for comparison. Leave as null for default behavior.

### Model configuration
After setting up the base-configuration file, the next step is to set up a configuration file for model under `configs/`.
#### API Model Configurations
This framework supports evaluating models using APIs such as OpenAI, Anthropic, Google, and Cohere. You need to create a separate config file for each API model. For example, the config file for OpenAI's gpt-4o-2024-05-13 would be named `configs/config-gpt-4o-2024-05-13.yaml`.

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **api:** Choose the API to use from `openai`, `anthropic`, `google`, `amazon_bedrock`.
- **batch_size:** Batch size for API calls (recommended: 32).
- **model:** Information about the model. 
    - `pretrained_model_name_or_path`: Name of the API model.
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (leave as null for API models).
    - `release_date`: Model release date. (MM/DD/YYYY)

#### Other Model Configurations

This framework also supports evaluating models using VLLM.  You need to create a separate config file for each VLLM model. For example, the config file for Microsoft's Phi-3-medium-128k-instruct would be named `configs/config-Phi-3-medium-128k-instruct.yaml`.

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **api:** Set to `vllm` to indicate using a VLLM model.
- **num_gpus:** Number of GPUs to use.
- **batch_size:** Batch size for VLLM (recommended: 256).
- **model:** Information about the model.
    - `artifacts_path`: When loading a model from wandb artifacts, it is necessary to include a description. If not, there is no need to write it. Example notation: wandb-japan/llm-leaderboard/llm-jp-13b-instruct-lora-jaster-v1.0:v0   
    - `pretrained_model_name_or_path`: Name of the VLLM model.
    - `chat_template`: Path to the chat template file (if needed).
    - `size_category`: Specify model size category. In Nejumi Leaderboard, the category is defined as "10B<", "10B<= <30B", "<=30B" and "api".
    - `size`: Model size (parameter).
    - `release_date`: Model release date (MM/DD/YYYY).
    - `max_model_len`: Maximum token length of the input (if needed).


#### Create Chat template (needed for models except for API)
1. create chat_templates/model_id.jinja
If the chat_template is specified in the tokenizer_config.json of the evaluation model, create a .jinja file with that configuration.
If chat_template is not specified in tokenizer_config.json, refer to the model card or other relevant documentation to create a chat_template and document it in a .jinja file.

2. test chat_templates
If you want to check the output of the chat_templates, you can use the following script:
```bash
python3 scripts/test_chat_template.py -m <model_id> -c <chat_template>
```
If the model ID and chat_template are the same, you can omit -c <chat_template>.


## Evaluation Execution
Once you prepare the dataset and the configuration files, you can run the evaluation process.

You can use either `-c` or `-s` option:
    - **-c (config):** Specify the config file by its name, e.g., `python3 scripts/run_eval.py -c config-gpt-4o-2024-05-13.yaml`
    - **-s (select-config):** Select from a list of available config files. This option is useful if you have multiple config files. 
   ```bash
   python3 scripts/run_eval.py -s
   or 
   python3 scripts/run_eval.py -c
   ```


The results of the evaluation will be logged to the specified W&B project.

## When you want to edit runs or add additional evaluation metrics
Please refer to [belend_run_configs/README.md](blend_run_configs/README.md).



## Contributing
Contributions to this repository is welcom. Please submit your suggestions via pull requests. Please note that we may not accept all pull requests.

## License
This repository is available for commercial use. However, please adhere to the respective rights and licenses of each evaluation dataset used.

## Contact
For questions or support, please concatct to contact-kr@wandb.com.
