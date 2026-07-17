"""Analysis tools for simulation data"""

from nbody_pipeline.analysis.particle_tracker import ParticleTracker
from nbody_pipeline.analysis.current_lagrangian import CurrentMassLagrangianProcessor
from nbody_pipeline.analysis.galactic_orbit import GalacticOrbitProcessor
from nbody_pipeline.analysis.galactic_energy_angular_momentum import (
    GalacticEnergyAngularMomentumProcessor,
)
from nbody_pipeline.analysis.compact_binary_counter import CompactBinaryCounter
from nbody_pipeline.analysis.b_type_binary import BTypeBinaryExtractor
from nbody_pipeline.analysis.binary_stellar_type import BinaryStellarTypeExtractor
from nbody_pipeline.analysis.primordial_binary import PrimordialBinaryIdentifier
from nbody_pipeline.analysis.initial_total_mass import InitialTotalMassAnalyzer
from nbody_pipeline.analysis.intermediate_mass_black_hole import (
    IntermediateMassBlackHoleAnalyzer,
)
from nbody_pipeline.analysis.compact_object_history import (
    CompactObjectHistoryProcessor,
    CompactObjectHistoryTask,
)
from nbody_pipeline.analysis.snapshot_summary import (
    SnapshotSummaryProcessor,
    SnapshotSummaryTask,
)
from nbody_pipeline.analysis.particle_lake import (
    ParticleLakeProcessor,
    SnapshotSinglesTask,
    SnapshotBinariesTask,
    SnapshotMergersTask,
    SnapshotScalarsTask,
)
from nbody_pipeline.analysis.physics import (
    tau_gw,
    compute_binary_orbit_relative_positions,
    compute_individual_orbit_params,
    binding_energy_nb,
    ebind_over_kt,
    TEMPORARY_EBIND_FACTOR,
    BINARY_CLASS_HARD,
    BINARY_CLASS_SOFT,
    BINARY_CLASS_TEMPORARY,
    BINARY_CLASS_ORDER,
    mean_core_interparticle_distance_au,
    classify_binaries,
    add_binary_energetics_and_class,
    drop_temporary_binaries,
)
from nbody_pipeline.analysis.kstar_semantics import (
    CmKstarState,
    decode_cm_kstar,
    decode_member_kw,
    annotate_binary_states,
)
from nbody_pipeline.analysis.bin_label_backfill import (
    ALGORITHM_VERSION as BIN_LABEL_BACKFILL_ALGORITHM_VERSION,
    reconstruct_bin_label,
    resolve_nzero,
    backfill_parts,
    validate_reconstruction,
)

__all__ = [
    "ParticleTracker",
    "CurrentMassLagrangianProcessor",
    "GalacticOrbitProcessor",
    "GalacticEnergyAngularMomentumProcessor",
    "CompactBinaryCounter",
    "BTypeBinaryExtractor",
    "BinaryStellarTypeExtractor",
    "PrimordialBinaryIdentifier",
    "InitialTotalMassAnalyzer",
    "IntermediateMassBlackHoleAnalyzer",
    "CompactObjectHistoryProcessor",
    "CompactObjectHistoryTask",
    "SnapshotSummaryProcessor",
    "SnapshotSummaryTask",
    "ParticleLakeProcessor",
    "SnapshotSinglesTask",
    "SnapshotBinariesTask",
    "SnapshotMergersTask",
    "SnapshotScalarsTask",
    "tau_gw",
    "compute_binary_orbit_relative_positions",
    "compute_individual_orbit_params",
    "binding_energy_nb",
    "ebind_over_kt",
    "TEMPORARY_EBIND_FACTOR",
    "BINARY_CLASS_HARD",
    "BINARY_CLASS_SOFT",
    "BINARY_CLASS_TEMPORARY",
    "BINARY_CLASS_ORDER",
    "mean_core_interparticle_distance_au",
    "classify_binaries",
    "add_binary_energetics_and_class",
    "drop_temporary_binaries",
    "CmKstarState",
    "decode_cm_kstar",
    "decode_member_kw",
    "annotate_binary_states",
    "BIN_LABEL_BACKFILL_ALGORITHM_VERSION",
    "reconstruct_bin_label",
    "resolve_nzero",
    "backfill_parts",
    "validate_reconstruction",
]
