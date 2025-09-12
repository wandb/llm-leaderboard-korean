from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ExternalProvider(ABC):
    """
    Generic contract for external evaluation/generation providers.
    Implementations wrap third-party pipelines (e.g., bfcl_eval) without changing them.
    """

    @abstractmethod
    def list_categories(self) -> Dict[str, List[str]]: ...

    @abstractmethod
    def list_models(self) -> List[str]: ...

    @abstractmethod
    def paths(self) -> Dict[str, Any]: ...

    @abstractmethod
    def generate(
        self,
        models: List[str],
        test_categories: Optional[List[str]] = None,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def evaluate(
        self,
        models: Optional[List[str]] = None,
        test_categories: Optional[List[str]] = None,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def load_scores(self, **kwargs) -> Dict[str, Any]: ...
