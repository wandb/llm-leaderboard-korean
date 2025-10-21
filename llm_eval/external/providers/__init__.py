# Import providers to trigger registration
from .bfcl import BFCLProvider  # noqa: F401
try:
    from .hallulens import *
except Exception:
    pass

# SWE-bench Verified does not use a provider class in this project.
# Guard the import so importing this package won't fail.
try:
    from .swe_bench_verified import SWEBenchVerifiedProvider  # type: ignore # noqa: F401
except Exception:
    # No provider class required; evaluation is dispatched via evaluator branching.
    pass
