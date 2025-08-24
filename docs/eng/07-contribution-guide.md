## LLM Evaluation Framework Contribution Guide

This guide provides detailed instructions for developers who want to contribute to the Haerae Evaluation Toolkit. Whether you are adding or improving features related to Datasets, Evaluators, Scaling methods, or Backend modules, this guide will help ensure consistency, smooth integration with the codebase, and easier maintenance and extension.

---

### General Principles

#### • Code Style and Consistency
- Follow the naming conventions, indentation, comment styles, and error-handling patterns used in the existing codebase (PEP8).
- Include clear docstrings or comments for every new class, function, and method, explaining their purpose, arguments, and return values.

#### • Documentation and Testing
- All contributions must be accompanied by proper documentation, including usage examples.
- Ensure that the module's purpose and usage are clearly communicated to help other developers and users.
- Write sufficient tests to ensure that changes do not break existing functionality.
- Include at least minimal unit tests for newly added features.

#### • Registry Pattern Compliance
- The dataset, evaluation, scaling_method, and backend modules use a decorator-based registry system for extensibility.
- When adding a new class, make sure to register it in the corresponding module:
  - e.g., `@register_dataset("name")` for datasets, `@register_evaluator("name")` for evaluators.

#### • Reusability and Modularity
- Each contribution should be as independent and reusable as possible.
- Separate concerns clearly (e.g., data loading, model inference, decoding/scaling, evaluation) to make them easily replaceable or upgradable within the pipeline.

---

### 2. Dataset Contribution Guide

#### • Inheriting and Implementing BaseDataset
- All new dataset classes must inherit from `BaseDataset`.
- The constructor should accept and store parameters such as `dataset_name`, `split`, `subset` (if needed), and additional parameters (e.g., HuggingFace options).
- Implement the `load()` method to return a list of samples in the format `{ "input": ..., "reference": ..., (additional fields) }`.
- For `subset`, implement logic to load full data if `None`, a specific part if it's a string, or merge multiple subsets if it's a list.

#### • Optional Methods
- `get_raw_samples()`: Returns raw data, useful for debugging/analysis.
- `info()`: Returns metadata such as sample count and subtask information.

#### • Registering in the Registry
- Use `@register_dataset("dataset_name")` to register the dataset.
- Can be called via CLI or other modules with `load_dataset("dataset_name")`.

#### • Tips
- Handle exceptions carefully (e.g., authentication issues, network errors).
- Clearly comment on field mappings, as different datasets may have different field names.

---

### 3. Evaluation Contribution Guide

#### • Inheriting and Implementing BaseEvaluator
- Create a new evaluator class by inheriting `BaseEvaluator`.
- Override methods like `prepare_prompt()` or `parse_prediction()` as needed.
- Implement `evaluate_predictions(samples)` to return metrics in the format `{ "metric_name": value, ... }`.
- `evaluate(data, model)` manages the overall evaluation flow (optionally includes `judge_batch()` for multi-models).

#### • Support for Custom CoT Parser
- Evaluator should allow passing in a CoT parser function.
- The parser function must return a tuple of `(chain_of_thought, final_answer)`.
- Document how the parser is used via comments.

#### • Registry Registration
- Register using `@register_evaluator("evaluator_name")`.

#### • Tips
- Refer to existing metric examples for consistency.
- Use logging to capture intermediate outputs (e.g., parsing results, scoring).

---

### 4. Scaling Method Contribution Guide

#### • Inheriting and Implementing BaseScalingMethod
- Inherit `BaseScalingMethod` to implement a new scaling strategy.
- Use `apply(data)` to apply the scaling method (e.g., beam search, best-of-N, majority voting).
- Implement `set_params()` if the method needs to dynamically update parameters.

#### • Registry Registration
- Register with `@register_scaling_method("scaling_method_name")`.

#### • Tips
- Clearly document selection logic (e.g., log-prob accumulation, pruning).
- Consider handling deduplication, EOS tokens, and other enhancements.

---

### 5. Backend Contribution Guide

#### • Inheriting and Implementing Model Interfaces
- Implement `BaseModel`, `BaseJudge`, or `BaseRewardModel` depending on your backend.
  - `BaseModel`: Must implement `generate_batch(inputs, return_logits=False)`.
  - `BaseJudge`: Must implement `judge_batch(inputs)`.
  - `BaseRewardModel`: Must implement `score_batch(inputs)`.

#### • Registry Registration
- Use `@register_model("backend_name")` to register.

#### • Tips
- For large models or external APIs, optimize for performance and error handling.
- Manage configurations via environment variables or config files, and document them clearly.
- Ensure compatibility with versions when using external libraries like HuggingFace.

---

### Additional Notes and Tips

#### • Custom CoT Parser
- Implement a function that returns `(chain_of_thought, final_answer)` tuple.
- Can be passed to Evaluator or PipelineRunner as `custom_cot_parser`.
- Via CLI, use the `--cot_parser` flag with a string like `package.module.function_name`.

#### • CLI and API Integration
- Make sure your code can be reused from both CLI and API.
- Use helper functions like `_parse_json_str()` to handle JSON string parameters.
- Expose the `PipelineRunner` and `Evaluator` structure cleanly through APIs.

#### • Testing and Debugging
- Always write unit and integration tests after feature implementation.
- Use logging for intermediate outputs at each stage (data loading, inference, evaluation).
- Collaborate via code review to improve and refine contributions.

---

### Final Note

Please review this guide before starting your contribution. If you have any questions or suggestions, contact the project maintainers (refer to `quick_start.md`).

### Dataset Contributions

If you are adding a new dataset, please follow the Dataset Development Guide:

- See: `docs/eng/09-dataset-development-guide.md`
- Keep the output schema consistent: at least `input`, `reference`, and include `_subset_name` when applicable.
- Prefer default `base_prompt_template` with an override via `dataset_params`.
- Add an example configuration under `examples/` when relevant.


