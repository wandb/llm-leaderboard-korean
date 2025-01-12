# Haerae-evaluation-toolkit
Haerae-evaluation-toolkit is an emerging open-source Python library designed to streamline and standardize the evaluation of Large Language Models (LLMs). 

## Our framework supports:

Multiple Evaluation Methods (Logit-Based, String-Match, LLM-as-a-Judge, etc.)  
Reasoning Chain Analysis for extended Korean chain-of-thought  
Various Korean Datasets (HAE-RAE Bench, KMMLU, KUDGE, QARV, K2-Eval, HRM8K)  
Scalable Inference-Time Techniques (Best-of-N, Majority Voting, Beam Search, etc.)  
OpenAI-Compatible Endpoints, vLLM, LiteLLM integration  
Flexible and Pluggable Architecture, allowing easy extension for new datasets, new evaluation metrics, and new inference backends.  

## Project Status
We are currently in active development, building out core features and interfaces. Our primary goals include:

Unified API for loading diverse Korean benchmark datasets.  
Configurable Inference Scaling (e.g., best-of-N, beam search) to generate higher-quality outputs for evaluation.  
Pluggable Evaluation Methods, providing chain-of-thought assessment, logit-based scoring, and standard metrics.  
Modular Architecture, enabling easy extension for additional backends, new tasks, or custom evaluation logic.  

## Key Features
**Dataset Abstraction**: Load and preprocess your dataset (or multiple subsets) with minimal configuration.  
**Scalable Methods**: Apply a variety of decoding strategies, from simple sampling to beam search or best-of-N approaches.  
**Evaluation Library**: Compare predictions against references, use separate judge models, or implement custom scoring.  
**Registry System**: Add new components (datasets, models, scaling methods) by registering them via decorators.  


## Contributing & Contact
We welcome collaborators, contributors, and testers who are interested in advancing LLM evaluation methods, especially in the context of Korean language tasks. If you would like to get involved, please reach out via one of the following emails:

gksdnf424@gmail.com  (Development Lead)  
spthsrbwls123@yonsei.ac.kr  (Research Lead)  
We look forward to hearing your ideas and contributions!  

## License  
Apache License 2.0  
© 2025 The HAE-RAE Team. All rights reserved.

