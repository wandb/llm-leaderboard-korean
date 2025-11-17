import os

lgai_configs = [
    # "EXAONE-4.0-1.2B",
    # "EXAONE-4.0.1-32B",
    # "EXAONE-4.0-32B"
]

openai_configs = [
    # "gpt-4o-2024-11-20",
    # "gpt-4.1-2025-04-14",
    "gpt-5-nano-2025-08-07",
    # "gpt-5-mini-2025-08-07",
    # "gpt-5-2025-08-07",
    # "o4-mini-2025-04-16",
]

anthropic_configs = [
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514"
]

xai_configs = [
    # "grok-4-fast-non-reasoning",
    # "grok-4-fast-reasoning",
    # "grok-4-0709",
]

google_configs = [
    "gemini-2.5-flash-lite",
    # "gemini-2.5-flash",
    # "gemini-2.5-pro",
]

print("\n\n-----------------------------------------------------")
print("-----------------------------------------------------")
print("Running evaluations for all specified model configurations...")
print("-----------------------------------------------------")
print("-----------------------------------------------------\n\n")
for config in lgai_configs+openai_configs+anthropic_configs+xai_configs+google_configs:#openai_configs + xai_configs:# anthropic_configs + openai_configs + xai_configs:
    print(f"||||   python run_eval.py --config {config}   ||||\n\n")
    os.system(f"python run_eval.py --config {config}")