# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-405B-Instruct-FP8"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    
)

# exp defualt = nonsense_all 
# Options: [nonsense_medicine nonsense_animal nonsense_plant nonsense_bacteria]

for SEED in 0
do
    for MODEL in "${MODELS[@]}"
    do
        python -m tasks.refusal_test.nonsense_mixed_entities \
            --exp nonsense_all \
            --do_generate_prompt \
            --tested_model $MODEL \
            --inference_method together \
            --seed $SEED
    done
done

#            --do_inference \
#            --do_eval \