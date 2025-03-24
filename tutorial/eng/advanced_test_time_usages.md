# 1. Best-of-N Technique

## 1.1 What is Best-of-N?
- **Best-of-N** is a method that samples the same input (prompt) multiple times (`N` times), then selects the response with the **highest score** (based on log probability, reward model, etc.) as the final output.
- Since even identical prompts may produce slightly different outputs due to internal model sampling, this method can significantly improve performance.

## 1.2 How to Use Best-of-N
```python
from llm_eval.evaluator import Evaluator

# 1) Create an Evaluator instance
evaluator = Evaluator()

# 2) Run evaluation using Best-of-N
results = evaluator.run(
    model="huggingface",  # or "openai", "vllm", etc.
    model_params={
        "model_name_or_path": "kakaocorp/kanana-nano-2.1b-instruct",
        "device": "cuda:0",
        "batch_size": 1,
        "max_new_tokens": 128
    },

    dataset="haerae_bench",
    subset=["csat_geo"],
    split="test",

    # Best-of-N configuration
    scaling_method="best_of_n",
    scaling_params={
        "n": 5,  # Number of samples per prompt
        "batch_size": 1
    },

    evaluation_method="string_match"  # Evaluation criterion
)

print(results)
df = results.to_dataframe()
print(df)
```

# 2. Beam Search Technique

## 2.1 What is Beam Search?
- Beam Search is an algorithm that keeps track of multiple candidate sequences (beams) during each generation step.
- Compared to Greedy Search (which maintains only one candidate at each step), Beam Search explores a broader range of possibilities and often produces more accurate results.
- Increasing beam_size expands the search range, but also increases computational costs and inference time. Finding the right balance is crucial.

## 2.2 How to Use Beam Search
```python
from llm_eval.evaluator import Evaluator

# 1) Create an Evaluator instance
evaluator = Evaluator()

# 2) Apply Beam Search
results = evaluator.run(
    model="huggingface",
    model_params={
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "device": "cuda:0",
        "batch_size": 1
    },

    dataset="haerae_bench",
    subset=["csat_geo"],
    split="test",

    scaling_method="beam_search",
    scaling_params={
        "beam_size": 4,      # Number of beams
        "max_tokens": 50,    # Max tokens to generate
        "agg_strategy": "sum",  # Score aggregation method (sum, mean, max, etc.)
        "batch_size": 1
    },

    evaluation_method="string_match"
)

print(results)
df = results.to_dataframe()
print(df)
```

# 3. Important Notes and Tips

1. **Time and Resource Usage**  
   - Both **Beam Search** and **Best-of-N** involve multiple inference runs or simultaneous exploration of several candidate sequences, which significantly **increases inference time** and GPU memory usage.
   - If faster results are needed, consider reducing the `beam_size` or the `n` parameter.

2. **Compatibility Between Scaling Methods**  
   - HRET allows the application of only **one `scaling_method`** per evaluation (`run()`) call.
   - For instance, it is **not possible** to simultaneously use both `best_of_n` and `beam_search`.

3. **Integration with LLM-as-Judge**  
   - When applying **Best-of-N** or **Beam Search**, instead of relying solely on log probabilities, you can leverage an **LLM-as-Judge** to score and select among candidate responses.
   - Specify a `judge_model` or `reward_model` to have the model directly evaluate and rank candidate responses.
   - This approach is particularly useful when more natural evaluation criteria or specific standards are required.



