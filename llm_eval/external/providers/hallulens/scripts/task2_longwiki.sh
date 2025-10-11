# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

GPU=0
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "meta-llama/Llama-3.1-405B-Instruct-FP8"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "google/gemma-2-9b-it"
    # "google/gemma-2-27b-it"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "mistralai/Mistral-Nemo-Instruct-2407"
    # "claude-3-sonnet"
    # "claude-3-haiku"
    # "gpt-4o"
#    "google/gemma-3n-E4B-it"
)

EXP_MODE=longwiki
for MODEL in "${MODELS[@]}"
do  
    CUDA_VISIBLE_DEVICES=$GPU python3 -m tasks.longwiki.longwiki_main \
        --exp_mode $EXP_MODE \
        --do_generate_prompt \
        --model $MODEL\
        --inference_method together \
        --N 250 \
        --db_path "/Users/jinminseong/Desktop/KoHalluLens/data/wiki_data/.cache/enwiki-20230401.db"
done

#        --do_inference \
#        --do_eval \
#        --N 3 \
#        --q_generator meta-llama/Llama-3.1-8B-Instruct    #TODO!!