"""
LLM Evaluation Toolkit

This package provides tools for evaluating Large Language Models (LLMs)
with support for various datasets, models, and evaluation methods.
"""

# Core components
from .runner import PipelineRunner, PipelineConfig

# HRET - MLOps-friendly facade API
from . import hret

# Version info
__version__ = "0.1.0"

# Main exports
__all__ = [
    "PipelineRunner",
    "PipelineConfig", 
    "hret",
]