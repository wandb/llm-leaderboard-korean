import os
for config in ["gpt-4o-2024-11-20", "gpt-4.1-2025-04-14", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "o4-mini-2025-04-16"]:
    os.system(f"python run_eval.py --config {config}")