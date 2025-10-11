# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

MODELS=(
#    "meta-llama/Llama-3.1-8B-Instruct"
#     "meta-llama/Llama-3.1-70B-Instruct"
#     "meta-llama/Llama-3.1-405B-Instruct-FP8"
#     "meta-llama/Llama-3.3-70B-Instruct"
    # "google/gemma-2-9b-it"
    # "google/gemma-2-27b-it"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "mistralai/Mistral-Nemo-Instruct-2407"
    # "claude-3-sonnet"
    # "claude-3-haiku"
    # "gpt-4o"
    "google/gemma-3n-E4B-it"
#    "marin-community/marin-8b-instruct"
)

MODE=dynamic
for MODEL in "${MODELS[@]}"
do
    python -m tasks.shortform.precise_wikiqa \
        --do_inference \
        --do_eval \
        --model $MODEL\
        --inference_method together
done
