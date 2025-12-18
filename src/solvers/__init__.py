"""
Custom Solvers

Custom solvers are imported and exposed here.
"""

from solvers.bfcl_solver import bfcl_solver, bfcl_text_solver
from solvers.mtbench_solver import mtbench_solver

__all__ = [
    "bfcl_solver",       # Native tool calling (OpenAI, Claude, Gemini, etc.)
    "bfcl_text_solver",  # Text-based (EXAONE, some open-source models, etc.)
    "mtbench_solver",    # MT-Bench 2-turn conversation
]
