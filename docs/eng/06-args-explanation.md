## 🔍 Evaluation Methods

### String / Partial Match Evaluation
Evaluates whether the model's prediction exactly matches or partially matches the reference.

#### 1. Text Normalization
Before comparing predictions and references or options, the following normalization options can be applied:
- **Ignore case** (`ignore_case=True`): Converts all text to lowercase for comparison.
- **Ignore punctuation** (`ignore_punctuation=False`): Optionally removes punctuation (.,!?).
- **Ignore numbers** (`ignore_numbers=False`): Optionally removes numeric characters.
- **Regex filtering** (`regexes_to_ignore`): Removes specific patterns using regular expressions.
- **Whitespace normalization**: Replaces multiple spaces with a single space.

#### 2. Final Answer Extraction
Extracts only the text after "Answer:" from the model output. This is useful when the output includes a reasoning chain.

#### 📖 Example
**Exact Match:**  
If the reference is "서울특별시" and the prediction is also "서울특별시", it is considered correct. In MCQA, if the options are ["서울", "부산", "대구"] and the prediction is "부산", it is also marked correct.

**Partial Match:**  
If the reference is "서울특별시" and the prediction is "대한민국 서울특별시 강남구", it's a partial match since the reference is included. In MCQA, if the prediction is "저는 서울에 가고 싶습니다.", the presence of "서울" is enough for a partial match.

---

### Log Probability Evaluation
Uses the log probabilities of generated options to determine the predicted answer and compares it with the reference.

#### 1. Input Structure
Samples are expected to include the following fields:
- `options`: List of options (e.g., ["Seoul", "Busan", "Daegu"])
- `logits`: Dictionary containing log probability information per option
- `option_log_probs`: List of log probabilities per option (e.g., [-1.2, -0.5, -2.3])
- `reference`: Correct answer option (e.g., "Busan")
- (Optional) `prediction`: Used if log-probability-based prediction isn't available

#### 2. Prediction by Log Probabilities
Selects the option with the highest log probability as the prediction:
- Example: With `option_log_probs = [-1.2, -0.5, -2.3]`, the second option "Busan" is selected
- **Fallback**: If no valid log probabilities are available, uses the `prediction` field instead

#### 3. Accuracy Calculation
Prediction is compared with the reference, and accuracy is computed as the proportion of correct predictions.

---

### Math Match Evaluation
Compares mathematical expressions for equivalence rather than exact string matching.

#### Key Features
- Checks if two expressions represent the same mathematical meaning
- Supports both LaTeX and plain math expressions
- Can extract final answers from reasoning chains using regex

#### 1. Input Structure
- `prediction`: Model's generated expression (e.g., "\boxed{1,2,3}")
- `reference`: Ground truth expression (e.g., "{1,2} \cup {3}")

#### 2. Final Answer Extraction
Uses regex patterns to extract the final answer:
- Example: "Answer: \boxed{1,2,3}" → "\boxed{1,2,3}"
- Regex patterns used:
  - `정답\s*:\s*(.*?)(?:\n|$)`
  - `Answer\s*:\s*(.*?)(?:\n|$)`

#### 3. Math Expression Parsing
Uses the `math_verify` library to parse expressions:
- Example: `\boxed{1,2,3}` → parsed into a math object
- If parsing fails, the sample is skipped and counted toward the `parse_failure_rate`

#### 4. Equivalence Check
Checks if parsed prediction and reference represent the same result:
- Example: `{1,2} \cup {3}` vs `{3,1,2}` → equivalent → marked correct
- Failures are counted in `verify_failure_rate`

#### 5. Accuracy Calculation
Returns the proportion of samples with successful equivalence verification.

##### 📖 Example
```python
samples = [
    {
        "prediction": "Answer: \boxed{1,2,3}",
        "reference": "{1,2} \cup {3}"
    },
    {
        "prediction": "Answer: x^2 + 2x + 1",
        "reference": "(x+1)^2"
    }
]
```

---

## Scaling Methods (Optional; may require long runtime)
- **self_consistency**: Generates multiple responses and selects the most frequent one
- **greedy**: Always picks the token with the highest probability (fast but lacks diversity)
- **beam_search**: Explores multiple candidate sequences to find the best one
- **top_k**: Randomly selects from the top-k most probable tokens
- **top_p**: Selects from a token pool where cumulative probability exceeds p
- **temperature_sampling**: Adjusts distribution for more creative outputs (higher temperature = more diverse)