# vLLM LiteLLM Configuration - Port Assignments

This document lists the port assignments for all Korean models in alphabetical order, starting from port 8100.

## Port Assignments

| Port | Model Name | HuggingFace Model | Config File |
|------|------------|-------------------|-------------|
| 8100 | A.X-3.1 | skt/A.X-3.1 | skt-a.x-3.1.yaml |
| 8101 | A.X-3.1-Light | skt/A.X-3.1-Light | skt-a.x-3.1-light.yaml |
| 8102 | A.X-4.0 | skt/A.X-4.0 | skt-a.x-4.0.yaml |
| 8103 | A.X-4.0-Light | skt/A.X-4.0-Light | skt-a.x-4.0-light.yaml |
| 8104 | EXAONE-3.0-7.8B-Instruct | LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct | lgai-exaone-exaone-3.0-7.8b-instruct.yaml |
| 8105 | EXAONE-3.5-2.4B-Instruct | LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct | lgai-exaone-exaone-3.5-2.4b-instruct.yaml |
| 8106 | EXAONE-3.5-7.8B-Instruct | LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct | lgai-exaone-exaone-3.5-7.8b-instruct.yaml |
| 8107 | EXAONE-3.5-32B-Instruct | LGAI-EXAONE/EXAONE-3.5-32B-Instruct | lgai-exaone-exaone-3.5-32b-instruct.yaml |
| 8108 | EXAONE-4.0-1.2B | LGAI-EXAONE/EXAONE-4.0-1.2B | lgai-exaone-exaone-4.0-1.2b.yaml |
| 8109 | EXAONE-4.0-32B | LGAI-EXAONE/EXAONE-4.0-32B | lgai-exaone-exaone-4.0-32b.yaml |
| 8110 | EXAONE-4.0.1-32B | LGAI-EXAONE/EXAONE-4.0.1-32B | lgai-exaone-exaone-4.0.1-32b.yaml |
| 8111 | HyperCLOVAX-SEED-Text-Instruct-1.5B | naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B | naver-hyperclovax-hyperclovax-seed-text-instruct-1.5b.yaml |
| 8112 | HyperCLOVAX-SEED-Think-14B | naver-hyperclovax/HyperCLOVAX-SEED-Think-14B | naver-hyperclovax-hyperclovax-seed-think-14b.yaml |
| 8113 | kanana-1.5-2.1b-instruct-2505 | kakaocorp/kanana-1.5-2.1b-instruct-2505 | kakaocorp-kanana-1.5-2.1b-instruct-2505.yaml |
| 8114 | kanana-1.5-8b-instruct-2505 | kakaocorp/kanana-1.5-8b-instruct-2505 | kakaocorp-kanana-1.5-8b-instruct-2505.yaml |
| 8115 | kanana-1.5-15.7b-a3b-instruct | kakaocorp/kanana-1.5-15.7b-a3b-instruct | kakaocorp-kanana-1.5-15.7b-a3b-instruct.yaml |
| 8116 | Midm-2.0-Base-Instruct | K-intelligence/Midm-2.0-Base-Instruct | k-intelligence-midm-2.0-base-instruct.yaml |
| 8117 | Midm-2.0-Mini-Instruct | K-intelligence/Midm-2.0-Mini-Instruct | k-intelligence-midm-2.0-mini-instruct.yaml |
| 8118 | SOLAR-10.7B-Instruct-v1.0 | upstage/SOLAR-10.7B-Instruct-v1.0 | upstage-solar-10.7b-instruct-v1.0.yaml |
| 8119 | solar-pro-preview-instruct | upstage/solar-pro-preview-instruct | upstage-solar-pro-preview-instruct.yaml |
| 8120 | Tri-7B | trillionlabs/Tri-7B | trillionlabs-tri-7b.yaml |
| 8121 | Tri-21B | trillionlabs/Tri-21B | trillionlabs-tri-21b.yaml |

### Reasoning Mode Configurations

| Port | Model Name | HuggingFace Model | Config File | Base Port | Notes |
|------|------------|-------------------|-------------|-----------|-------|
| 9108 | EXAONE-4.0-1.2B (Reasoning) | LGAI-EXAONE/EXAONE-4.0-1.2B | lgai-exaone-exaone-4.0-1.2b_reasoning.yaml | 8108 | `enable_thinking: true` |
| 9109 | EXAONE-4.0-32B (Reasoning) | LGAI-EXAONE/EXAONE-4.0-32B | lgai-exaone-exaone-4.0-32b_reasoning.yaml | 8109 | `enable_thinking: true` |
| 9110 | EXAONE-4.0.1-32B (Reasoning) | LGAI-EXAONE/EXAONE-4.0.1-32B | lgai-exaone-exaone-4.0.1-32b_reasoning.yaml | 8110 | `enable_thinking: true` |
| 9112 | HyperCLOVAX-SEED-Think-14B (Reasoning) | naver-hyperclovax/HyperCLOVAX-SEED-Think-14B | naver-hyperclovax-hyperclovax-seed-think-14b_reasoning.yaml | 8112 | `force_reasoning: true` |

## Notes

- Port assignments follow the alphabetical order of model names as defined in `llm_eval/external/providers/bfcl/bfcl_eval/constants/supported_models.py`
- Each YAML file has two port-related fields updated:
  - `model.params.api_base`: The API endpoint URL (e.g., `http://localhost:8100/v1`)
  - `model.vllm_params.port`: The port number (e.g., `8100`)
- EXAONE-Deep models (2.4B, 7.8B, 32B) are not included as they don't have corresponding configuration files
- **Standard models**: Port range 8100-8121 (22 models total)
- **Reasoning mode models**: Port range 9100-9199 (4 models total)
  - Reasoning ports follow the pattern: Base port 8xxx → Reasoning port 9xxx
  - Example: EXAONE-4.0-1.2B uses 8108, its reasoning mode uses 9108
- **Total models: 26 configurations**
- Reasoning mode configurations use `extra_body.chat_template_kwargs` to enable thinking capabilities