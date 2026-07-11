"""
nbody-pipeline - N-body simulation data analysis and visualization toolkit
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.10.0"

if TYPE_CHECKING:
    from nbody_pipeline.__main__ import SimulationPlotter, main
    from nbody_pipeline.analysis import (
        BTypeBinaryExtractor,
        BinaryStellarTypeExtractor,
        InitialTotalMassAnalyzer,
        IntermediateMassBlackHoleAnalyzer,
        PrimordialBinaryIdentifier,
    )


def __getattr__(name: str) -> Any:
    """Lazily expose CLI entry points without importing __main__ during -m startup."""
    if name in {"main", "SimulationPlotter"}:
        from nbody_pipeline.__main__ import SimulationPlotter, main

        exported = {"main": main, "SimulationPlotter": SimulationPlotter}
        globals().update(exported)
        return exported[name]
    if name in {
        "BTypeBinaryExtractor",
        "BinaryStellarTypeExtractor",
        "InitialTotalMassAnalyzer",
        "IntermediateMassBlackHoleAnalyzer",
        "PrimordialBinaryIdentifier",
    }:
        from nbody_pipeline.analysis import (
            BTypeBinaryExtractor,
            BinaryStellarTypeExtractor,
            InitialTotalMassAnalyzer,
            IntermediateMassBlackHoleAnalyzer,
            PrimordialBinaryIdentifier,
        )

        exported = {
            "BTypeBinaryExtractor": BTypeBinaryExtractor,
            "BinaryStellarTypeExtractor": BinaryStellarTypeExtractor,
            "InitialTotalMassAnalyzer": InitialTotalMassAnalyzer,
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
    "InitialTotalMassAnalyzer",
    "IntermediateMassBlackHoleAnalyzer",
    "PrimordialBinaryIdentifier",
]
