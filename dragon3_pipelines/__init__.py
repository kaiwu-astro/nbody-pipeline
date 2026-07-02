"""
Dragon3 Pipelines - N-body simulation data analysis and visualization toolkit
"""

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

if TYPE_CHECKING:
    from dragon3_pipelines.__main__ import SimulationPlotter, main
    from dragon3_pipelines.analysis import (
        BTypeBinaryExtractor,
        BinaryStellarTypeExtractor,
        IntermediateMassBlackHoleAnalyzer,
        PrimordialBinaryIdentifier,
    )


def __getattr__(name: str) -> Any:
    """Lazily expose CLI entry points without importing __main__ during -m startup."""
    if name in {"main", "SimulationPlotter"}:
        from dragon3_pipelines.__main__ import SimulationPlotter, main

        exported = {"main": main, "SimulationPlotter": SimulationPlotter}
        globals().update(exported)
        return exported[name]
    if name in {
        "BTypeBinaryExtractor",
        "BinaryStellarTypeExtractor",
        "IntermediateMassBlackHoleAnalyzer",
        "PrimordialBinaryIdentifier",
    }:
        from dragon3_pipelines.analysis import (
            BTypeBinaryExtractor,
            BinaryStellarTypeExtractor,
            IntermediateMassBlackHoleAnalyzer,
            PrimordialBinaryIdentifier,
        )

        exported = {
            "BTypeBinaryExtractor": BTypeBinaryExtractor,
            "BinaryStellarTypeExtractor": BinaryStellarTypeExtractor,
            "IntermediateMassBlackHoleAnalyzer": IntermediateMassBlackHoleAnalyzer,
            "PrimordialBinaryIdentifier": PrimordialBinaryIdentifier,
        }
        globals().update(exported)
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Public API
__all__ = [
    "__version__",
    "main",
    "SimulationPlotter",
    "BTypeBinaryExtractor",
    "BinaryStellarTypeExtractor",
    "IntermediateMassBlackHoleAnalyzer",
    "PrimordialBinaryIdentifier",
]
