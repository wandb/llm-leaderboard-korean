from __future__ import annotations

from typing import Any, List, Tuple


class LLMAsyncProcessor:
    """
    Minimal async-like processor wrapper used by swe_bench evaluator.
    Accepts a list of inputs [[messages, gen_kwargs], ...] and a callable llm(messages, **kwargs)
    that returns an object with .content and optionally .tool_calls.
    We run sequentially for simplicity to avoid adding new deps.
    """

    def __init__(self, llm, inputs: List[Tuple[Any, Any]]):
        self.llm = llm
        self.inputs = inputs

    def get_results(self) -> List[Any]:
        results: List[Any] = []
        for messages, gen_kwargs in self.inputs:
            try:
                # llm should accept (messages, **kwargs)
                out = self.llm(messages, **(gen_kwargs or {}))
            except TypeError:
                # fallback: some llm might accept (messages) only
                out = self.llm(messages)
            results.append(out)
        return results


