# Haerae-Evaluation-Toolkit

**Haerae-Evaluation-Toolkit** is an emerging open-source Python library designed to streamline and standardize the evaluation of **Large Language Models (LLMs)**.

---

## ✨ Key Features

- **Multiple Evaluation Methods**  
  Logit-Based, String-Match, LLM-as-a-Judge, and more.
  
- **Reasoning Chain Analysis**  
  Dedicated to analyzing extended Korean chain-of-thought reasoning.
  
- **Extensive Korean Datasets**  
  Includes HAE-RAE Bench, KMMLU, KUDGE, QARV, K2-Eval, and HRM8K.
  
- **Scalable Inference-Time Techniques**  
  Best-of-N, Majority Voting, Beam Search, and other advanced methods.
  
- **Integration-Ready**  
  Supports OpenAI-Compatible Endpoints, vLLM, and LiteLLM.

- **Flexible and Pluggable Architecture**  
  Easily extend with new datasets, evaluation metrics, and inference backends.

---

## 🚀 Project Status

We are actively developing core features and interfaces. Current goals include:

- **Unified API**  
  Seamless loading and integration of diverse Korean benchmark datasets.

- **Configurable Inference Scaling**  
  Generate higher-quality outputs through techniques like best-of-N and beam search.

- **Pluggable Evaluation Methods**  
  Enable chain-of-thought assessments, logit-based scoring, and standard evaluation metrics.

- **Modular Architecture**  
  Easily extendable for new backends, tasks, or custom evaluation logic.

---

## 🛠️ Key Components

### **Dataset Abstraction**
Load and preprocess your datasets (or subsets) with minimal configuration.

### **Scalable Methods**
Apply decoding strategies such as sampling, beam search, and best-of-N approaches.

### **Evaluation Library**
Compare predictions to references, use judge models, or create custom scoring methods.

### **Registry System**
Add new components (datasets, models, scaling methods) via simple decorator-based registration.

---

## 🤝 Contributing & Contact

We welcome collaborators, contributors, and testers interested in advancing LLM evaluation methods, especially for Korean language tasks.

📩 **Contact Us**:
- **Development Lead**: [gksdnf424@gmail.com](mailto:gksdnf424@gmail.com)  
- **Research Lead**: [spthsrbwls123@yonsei.ac.kr](mailto:spthsrbwls123@yonsei.ac.kr)

We look forward to hearing your ideas and contributions!

---

## 📜 License

Licensed under the **Apache License 2.0**.  
© 2025 The HAE-RAE Team. All rights reserved.
