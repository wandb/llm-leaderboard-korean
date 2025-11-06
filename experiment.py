import os
for config in ["gpt-5-mini-2025-08-07", "gemini-2.5-flash", "gpt-4.1-2025-04-14", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "o4-mini-2025-04-16", "grok-4-fast-non-reasoning"]:
    os.system(f"python run_eval.py --config {config}")