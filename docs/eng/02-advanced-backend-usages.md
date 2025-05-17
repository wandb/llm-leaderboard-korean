# Advanced Backend Usages

In this tutorial, we will explore advanced methods for utilizing the Haerae Evaluation Toolkit using LiteLLM and OpenAI-compatible APIs.

## 1. LiteLLM Integration

### 1.1 What is LiteLLM?
- **LiteLLM** is a library that provides a unified interface for various LLM providers (OpenAI, Azure, Anthropic, Claude, Cohere, etc.).
- It enables access to more than 30 LLM providers through a single interface, making it easy to switch between models.
- Haerae supports easy access to various models through LiteLLM.

### 1.2 LiteLLM Setup and Usage

```python
from llm_eval.evaluator import Evaluator

# 1) Create an Evaluator instance
evaluator = Evaluator()

# 2) Model evaluation through LiteLLM
results = evaluator.run(
    # Specify LiteLLM backend
    model="litellm",
    
    # LiteLLM parameter settings
    model_params={
        "model_name": "gpt-4", # OpenAI model name
        # Or other provider models like "anthropic/claude-3-opus-20240229"
        "api_key": "your-api-key-here",
        "max_tokens": 512,
        "temperature": 0.7
    },
    
    dataset="haerae_bench",
    subset=["csat_math"],
    split="test",
    
    evaluation_method="string_match"
)

print(results)
df = results.to_dataframe()
print(df)
```

### 1.3 Comparing Multiple Models

The advantage of LiteLLM is that you can easily compare multiple models using the same interface.

```python
# Multiple model configuration
models = [
    {
        "model": "litellm",
        "model_params": {
            "model_name": "gpt-4",
            "api_key": "your-openai-api-key"
        },
        "name": "GPT-4"  # Alias for result comparison
    },
    {
        "model": "litellm",
        "model_params": {
            "model_name": "anthropic/claude-3-opus-20240229",
            "api_key": "your-anthropic-api-key"
        },
        "name": "Claude-3"  # Alias for result comparison
    }
]

# Run model comparison evaluation
comparison_results = evaluator.compare(
    models=models,
    dataset="haerae_bench",
    subset=["csat_math", "csat_geo"],
    split="test",
    evaluation_method="string_match"
)

print(comparison_results)
print(comparison_results.to_dataframe())
```

## 2. Using OpenAI-Compatible APIs

### 2.1 What are OpenAI-Compatible APIs?
- Various LLM services provide endpoints that are compatible with OpenAI's API interface.
- This allows you to switch to other compatible services without modifying your code that uses the OpenAI client.
- Haerae provides methods to access these OpenAI-compatible endpoints.

### 2.2 Using OpenAI-Compatible APIs

```python
from llm_eval.evaluator import Evaluator

# 1) Create an Evaluator instance
evaluator = Evaluator()

# 2) Evaluation through OpenAI-Compatible API
results = evaluator.run(
    # Specify OpenAI backend
    model="openai",
    
    # OpenAI-Compatible API parameters
    model_params={
        "model_name": "text-davinci-003",  # Model name
        "api_key": "your-api-key-here",
        "api_base": "https://your-custom-endpoint.com/v1",  # Custom endpoint
        "max_tokens": 256,
        "temperature": 0.2
    },
    
    dataset="haerae_bench",
    subset=["csat_eng"],
    split="test",
    
    evaluation_method="string_match"
)

print(results)
df = results.to_dataframe()
print(df)
```

### 2.3 Using Self-Hosted OpenAI-Compatible Servers

When using self-hosted LLM servers (e.g., vLLM, FastChat, text-generation-inference, etc.):

```python
# Connect to a self-hosted OpenAI-compatible server
results = evaluator.run(
    model="openai",
    model_params={
        "model_name": "local-llama-2-13b",  # Local model name
        "api_base": "http://localhost:8000/v1",  # Local server address
        "api_key": "not-needed",  # Self-hosted servers may not require an API key
        "max_tokens": 512
    },
    
    dataset="haerae_bench",
    subset=["csat_kor"],
    split="test",
    
    evaluation_method="string_match"
)
```

## 3. Using HRET Agents

### 3.1 What are HRET Agents?
- **HRET Agents** is a tool that automates dataset preparation for the Haerae Evaluation Toolkit.
- It translates Hugging Face datasets into Korean and automatically generates dataset modules usable in HRET.
- Its modular design efficiently handles each step of the dataset preparation process.

### 3.2 HRET Agents Installation and Environment Setup

```bash
# Clone the HRET Agents repository
git clone https://github.com/HAE-RAE/hret-agents.git
cd hret-agents

# Install required packages
pip install -r requirements.txt
```

Configure API keys and settings in the `config/config.py` file:

```python
# config/config.py example
OPENAI_API_KEY = "your-openai-api-key-here"
HF_TOKEN = "your-huggingface-token-here"
BATCH_SIZE = 10  # Translation batch size
MAX_RETRIES = 3  # Number of retries on error
```

### 3.3 Basic Usage of HRET Agents

```bash
# Basic usage: Convert a Hugging Face dataset and save locally
python src/main.py --dataset "HAERAE-HUB/QARV" --subset "your_subset" --split train

# When automatically uploading to Hugging Face Hub
python src/main.py --dataset "HAERAE-HUB/QARV" --subset "your_subset" --split train --push
```

How to use directly in a Python script:

```python
from hret_agents.agent import DatasetTranslationAgent
from hret_agents.tools import (
    DatasetDownloader, 
    DatasetTranslator, 
    ModuleGenerator, 
    DatasetPusher
)

# Agent setup
agent = DatasetTranslationAgent(
    openai_api_key="your-openai-api-key",
    hf_token="your-huggingface-token"
)

# Add necessary tools
agent.add_tool(DatasetDownloader())
agent.add_tool(DatasetTranslator(batch_size=10, max_retries=3))
agent.add_tool(ModuleGenerator())
agent.add_tool(DatasetPusher())

# Execute dataset processing
result = agent.process_dataset(
    dataset_name="HAERAE-HUB/QARV",
    subset="your_subset",
    split="train",
    push_to_hub=True
)

print(f"Generated dataset module: {result['module_path']}")
```

### 3.4 Detailed Explanation of the Dataset Conversion Process

HRET Agents processes datasets in the following steps:

1. **Dataset Download**: Downloads the specified dataset from the Hugging Face Hub.
2. **Dataset Analysis**: Analyzes the dataset structure and converts the top 5 rows into a markdown table.
3. **Korean Translation**: Translates column names and content of the dataset into Korean (includes batch processing and retry logic).
4. **Module Generation**: Uses the OpenAI API to generate HRET-compatible dataset module code.
5. **Save and Upload**: Saves the generated module locally and optionally uploads it to the Hugging Face Hub.

### 3.5 Using Custom Dataset Guide Prompts

You can customize the guide prompt used for dataset module generation:

```python
# Define a custom guide prompt
custom_guide_prompt = """
You are an expert at creating dataset modules for the Haerae Evaluation Toolkit.
Please write a complete Python module based on the following dataset schema and sample data:

Dataset name: {dataset_name}
Dataset structure:
{dataset_schema}

Sample data:
{sample_data}

Please implement by inheriting the following BaseDataset class:
{base_class_definition}

Additional requirements:
1. All fields should be translated to Korean.
2. The get_prompt method should effectively format the problem.
3. The evaluate_prediction method should accurately evaluate the answer.
"""

# Module generation using a custom guide prompt
from hret_agents.tools import ModuleGenerator

module_generator = ModuleGenerator(guide_prompt=custom_guide_prompt)
module_code = module_generator.generate_module(
    dataset_info={
        "name": "HAERAE-HUB/QARV",
        "schema": "...",  # Dataset schema information
        "sample_data": "..."  # Sample data
    }
)

print(module_code)
```

## 4. Understanding HRET Dataset Module Structure

### 4.1 What is an HRET Dataset Module?
- An HRET dataset module is a standardized dataset class that can be used for evaluation in the Haerae Evaluation Toolkit.
- It is implemented by inheriting the `BaseDataset` class and provides functions for data loading, prompt generation, answer evaluation, etc.
- HRET Agents automates the generation of these modules, greatly reducing manual work.

### 4.2 Basic Structure of a Dataset Module

```python
# Example of an automatically generated dataset module
from llm_eval.dataset.base import BaseDataset
from datasets import load_dataset
import re

class QARVDataset(BaseDataset):
    def __init__(self, subset="your_subset", split="train"):
        """
        Initialize QARV dataset
        
        Args:
            subset (str): Subset to use
            split (str): Data split (train, validation, test)
        """
        self.subset = subset
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess dataset"""
        dataset = load_dataset("HAERAE-HUB/QARV", self.subset)
        self.data = dataset[self.split]
    
    def get_prompt(self, index):
        """
        Generate a prompt for a specific index
        
        Args:
            index (int): Data index
            
        Returns:
            str: Formatted prompt
        """
        example = self.data[index]
        prompt = f"Question: {example['question']}\n\n"
        prompt += f"Choose the correct answer from the following:\n"
        
        for i, option in enumerate(example['options']):
            prompt += f"{chr(65+i)}. {option}\n"
        
        return prompt
    
    def evaluate_prediction(self, index, prediction):
        """
        Evaluate prediction result
        
        Args:
            index (int): Data index
            prediction (str): Model's prediction result
            
        Returns:
            tuple: (correctness, score)
        """
        example = self.data[index]
        correct_answer = example['answer']
        
        # Extract answer like A, B, C, D from prediction
        pattern = r'([A-D])'
        matches = re.findall(pattern, prediction)
        
        if not matches:
            return False, 0
        
        predicted_answer = matches[0]
        is_correct = (predicted_answer == correct_answer)
        
        return is_correct, 1 if is_correct else 0
```

## 5. Dataset Generation and Usage Workflow

Let's explore a complete workflow utilizing both HRET Agents and LiteLLM/OpenAI-compatible API.

```python
import os
from llm_eval.evaluator import Evaluator
from hret_agents.agent import DatasetTranslationAgent
from hret_agents.tools import DatasetDownloader, DatasetTranslator, ModuleGenerator

# 1. Environment setup
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["HF_TOKEN"] = "your-huggingface-token"

# 2. Dataset preparation (using HRET Agent)
agent = DatasetTranslationAgent(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    hf_token=os.environ["HF_TOKEN"]
)

# Add necessary tools
agent.add_tool(DatasetDownloader())
agent.add_tool(DatasetTranslator(batch_size=10, max_retries=3))
agent.add_tool(ModuleGenerator())

# Execute dataset processing
result = agent.process_dataset(
    dataset_name="HAERAE-HUB/QARV",
    subset="your_subset",
    split="train"
)

# 3. Evaluation using LiteLLM
evaluator = Evaluator()

results = evaluator.run(
    model="litellm",
    model_params={
        "model_name": "gpt-4",
        "api_key": os.environ["OPENAI_API_KEY"],
        "max_tokens": 512
    },
    
    # Use the generated dataset module
    dataset="HAERAE-HUB/QARV",
    subset=["your_subset"],
    split="train",
    
    evaluation_method="string_match"
)

print(results)
print(results.to_dataframe())
```

## 6. Cautions and Tips

1. **API Key Security**
   - API keys should be managed securely. It's good to manage keys through environment variables or security settings.
   - Example: `os.environ.get("OPENAI_API_KEY")` or using a `.env` file

2. **Translation Cost and Time Management**
   - Translating large datasets can incur significant OpenAI API costs.
   - Optimize costs and time by adjusting batch size and retry counts appropriately.
   - For testing purposes, it's good to try with a small dataset subset first.

3. **Dataset Module Validation**
   - Automatically generated dataset modules should always be manually reviewed to verify quality.
   - In particular, check that the evaluation logic in the `evaluate_prediction` method is accurate.

4. **Switching Between Backends**
   - When switching between different backends, be aware that models may have different characteristics (token limits, prompt formats, etc.).
   - It's good to run the same test on multiple backends to compare results.

5. **Error Handling and Logging**
   - While HRET Agents includes retry logic for translation failures by default, additional error handling may be needed.
   - Use the `verbose=True` option to activate detailed logs for troubleshooting.

6. **Dataset Extension**
   - For new evaluation criteria, you can extend the generated dataset module to implement additional methods.
   - For example, you can add various evaluation metrics (BLEU, ROUGE, etc.) to evaluate performance from multiple angles.
