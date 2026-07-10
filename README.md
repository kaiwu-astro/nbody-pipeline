# Dragon3 Pipelines - N-body Simulation Data Analysis

A modular Python package for analyzing and visualizing N-body simulation data from Dragon3 simulations.

## Installation

### For Users

Clone this repository and install the package:

```bash
git clone https://github.com/kaiwu-astro/dragon3_pipeline.git
cd dragon3_pipeline
pip install -e .
```

### For Developers

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the shared CI test entrypoint:

```bash
./scripts/ci.sh
```

The CI script runs pytest in parallel with pytest-xdist, using at most 8 workers.

### Reproducing an Exact Environment

`requirements.lock` is a pinned snapshot (`pip freeze --exclude-editable`) of the
environment used to produce the results in a given release, refreshed automatically
by `scripts/release.sh`. It is not an installation constraint (use the looser
dependency ranges in `pyproject.toml`/`pip install -e .` for that) — use it when you
need to reproduce results bit-for-bit:

```bash
pip install -r requirements.lock
```

## Quick Start

### Using the Command Line

```bash
# Show available commands and options
python -m dragon3_pipelines --help

# Run with default configuration
python -m dragon3_pipelines

# Resume from existing plots
python -m dragon3_pipelines --skip-until=last

# Show purge command help
python -m dragon3_pipelines help purge

# Create movies from all preset plot patterns
bash dragon3_pipelines/scripts/dragon3_jpg_to_movie.sh

# Create movies for selected or custom plot patterns
bash dragon3_pipelines/scripts/dragon3_jpg_to_movie.sh create _CMD.jpg _custom_suffix.jpg

# Show movie command help and preset plot patterns
bash dragon3_pipelines/scripts/dragon3_jpg_to_movie.sh --help
bash dragon3_pipelines/scripts/dragon3_jpg_to_movie.sh create help
```

### Using as a Python Package

```python
from dragon3_pipelines import main, SimulationPlotter
from dragon3_pipelines.config import ConfigManager
from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.analysis import InitialTotalMassAnalyzer, ParticleTracker
from dragon3_pipelines.visualization import SingleStarVisualizer

# Create custom configuration
config = ConfigManager(config_path="my_config.yaml")  # see Configuration below
config.processes_count = 20

# Use visualizers
visualizer = SingleStarVisualizer(config)

# Read the cached initial total mass from lagr.7
initial_mass_msun = InitialTotalMassAnalyzer(config).get_initial_total_mass_msun("20sb")
```

## Configuration

The packaged default config (`dragon3_pipelines/config/default_config.yaml`) ships
**no site-specific paths** — `paths.simulations`, `paths.plot_dir`, and
`paths.analysis_cache_dir` are empty/`null` out of the box. Constructing a
`ConfigManager` without a user config that fills these in raises a clear
`ValueError` naming exactly which keys are missing. Everything else (physical
constants, stellar-type tables, plot limits, ...) has working scientific defaults
and does not need to be overridden.

### Providing a User Config

Copy [`config.example.yaml`](config.example.yaml) (a fully annotated template) and
fill in your paths, then point the CLI at it, in priority order:

```bash
python -m dragon3_pipelines --config /path/to/my_config.yaml
# or
export DRAGON3_CONFIG=/path/to/my_config.yaml
python -m dragon3_pipelines
# or drop it at ./dragon3_config.yaml in the current directory
```

`purge --list-targets` and `--help`/`help` work without any config. The main
pipeline and the `purge`/`analyze` subcommands need a config supplying at least
`paths.simulations`, `paths.plot_dir`, and `paths.analysis_cache_dir`.

madnuc/JUWELS users can use the tracked site config directly:

```bash
export DRAGON3_CONFIG=configs/juwels_madnuc.yaml
```

### Loading a Config in Code

```python
from dragon3_pipelines.config import load_config
config = load_config("my_config.yaml")
```

## Package Structure

```
dragon3_pipelines/
├── config/          # Configuration management
├── io/              # Data I/O (HDF5, text files)
├── analysis/        # Particle tracking, physics calculations
├── visualization/   # Plotting and visualization
├── utils/           # Utility functions
└── scripts/         # Shell scripts for movie generation
```

## Features

- **Modular Design**: Clean separation of I/O, analysis, and visualization
- **Type Annotations**: Full type hints for better IDE support
- **Backward Compatible**: All old scripts and imports still work
- **Configurable**: YAML-based configuration with sensible defaults
- **Parallel Processing**: Multi-process support for analyzing large datasets
- **Comprehensive Testing**: 250+ unit tests covering all modules

## Usage Examples

### Analyze HDF5 Files

```python
from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager(config_path="my_config.yaml")  # see Configuration below
processor = HDF5FileProcessor(config)
df_dict = processor.read_file(hdf5_path="/path/to/file.h5part", simu_name="my_sim")
# df_dict contains 'scalars', 'singles', 'binaries', 'mergers' DataFrames
```

### Track Particles

```python
from dragon3_pipelines.analysis import ParticleTracker
from dragon3_pipelines.config import ConfigManager

config = ConfigManager(config_path="my_config.yaml")  # see Configuration below
tracker = ParticleTracker(config)
particle_history = tracker.update_one_particle_history_df(simu_name="my_sim", particle_name=12345)
```

### Extract Binaries by Stellar Type

```python
from dragon3_pipelines.analysis import BinaryStellarTypeExtractor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager(config_path="my_config.yaml")  # see Configuration below
extractor = BinaryStellarTypeExtractor(config)
bh_binaries = extractor.load_binaries_with_stellar_type("my_sim", stellar_type="BH")
ns_binaries = extractor.load_binaries_with_stellar_type("my_sim", kw=13)
```

The returned table contains the complete processed binary rows for every snapshot where either binary component matches the requested stellar type or KW code.

### Analyze IMBH Candidates

```python
from dragon3_pipelines.analysis import IntermediateMassBlackHoleAnalyzer

imbh = IntermediateMassBlackHoleAnalyzer(config)
result = imbh.summarize_simulation("my_sim")
objects = result["objects"]
snapshots = result["snapshots"]
merger_events = result["merger_events"]
```

IMBH candidates are black holes with `100 < mass < 1e5` solar mass. The scan caches per-snapshot candidate rows under `intermediate_mass_black_hole`; linked physical merger events come from concatenated `coll.13` and `coal.24` files and are cached as true merger events. The HDF5 `mergers` table is not used for IMBH physical lineage.

HDF5 data-reduction tasks for getting macroscopic data can be batched with `HDF5ScanSession` so compatible tasks share HDF5 reads:
(here macroscopic data from HDF5 files means a few data points per HDF5 snapshot, such as center-of-mass info, largrangian data; in contract with many data points per snapshot such as X and V of all single stars)

```python
from dragon3_pipelines.analysis import BTypeBinaryExtractor, BinaryStellarTypeExtractor
from dragon3_pipelines.analysis.hdf5_scan import HDF5ScanSession

session = HDF5ScanSession(config)
session.add_job(BinaryStellarTypeExtractor(config).build_scan_job("my_sim", stellar_type="BH"))
session.add_job(BTypeBinaryExtractor(config).build_scan_job("my_sim"))
results = session.run()
```

### Plot the Cluster Galactic Orbit

```python
from dragon3_pipelines.analysis import GalacticOrbitProcessor
from dragon3_pipelines.visualization import GalacticOrbitVisualizer

orbit_df = GalacticOrbitProcessor(config).load_plot_data("my_sim")
viz = GalacticOrbitVisualizer(config)
viz.create_projection_plot(orbit_df, "my_sim")
viz.create_interactive_3d_html(orbit_df, "my_sim")
```

This scan-backed analysis caches scalar snapshot rows under `galactic_orbit` and plots the cluster position columns `RG(1..3)` colored by `Time[Myr]`.

### Plot Snapshot Galactic Energy vs Angular Momentum

```python
from dragon3_pipelines.analysis import GalacticEnergyAngularMomentumProcessor
from dragon3_pipelines.visualization import SingleStarVisualizer

plot_df = GalacticEnergyAngularMomentumProcessor(config).compute_snapshot(
    single_df_at_t,
    scalar_row_at_t,
)
SingleStarVisualizer(config).create_galactic_energy_angular_momentum_plot_jpg(plot_df, "my_sim")
```

The default HDF5 plotting flow creates `jpg/{prefix}output_ttot_{ttot}_galactic_E_vs_Lz.jpg` when `galactic_energy_angular_momentum.enabled` is true. The calculation uses `galpy` `MWPotential2014` with scalar `RG/VG` interpreted as pc/km/s and plots finite stellar points with axis limits from `percentile_limits`.

### Create Visualizations

```python
from dragon3_pipelines.visualization import BinaryStarVisualizer
from dragon3_pipelines.config import ConfigManager

config = ConfigManager(config_path="my_config.yaml")  # see Configuration below
viz = BinaryStarVisualizer(config)
# Get binary data at a specific time first
binary_df_at_t = df_dict['binaries'][df_dict['binaries']['TTOT'] == 100.0]
viz.create_mass_ratio_m1_plot_density(binary_df_at_t, simu_name="my_sim")
```

## Command Line Options

- `-h`, `--help`, `help`: Show top-level command help
- `help purge`: Show purge command help
- `--skip-until=N`: Start processing from time N
- `--skip-until=last`: Resume from last processed time
- `--config=PATH`: Path to a user config YAML (see [Configuration](#configuration))
- `--debug`: Enable debug logging

The installed `dragon3-plot` script accepts the same arguments as `python -m dragon3_pipelines`.

### Purge Generated Plots

```bash
# List supported purge targets
python -m dragon3_pipelines purge --list-targets

# Preview matching files before deleting
python -m dragon3_pipelines purge single.create_position_plot_jpg --simu sim_a --dry-run
python -m dragon3_pipelines purge single.create_galactic_energy_angular_momentum_plot_jpg --simu sim_a --dry-run

# Delete matching files without an interactive confirmation
python -m dragon3_pipelines purge single.create_position_plot_jpg --simu sim_a --yes
```

## Versioning & Changelog

This project follows [Semantic Versioning](https://semver.org/); `MAJOR.MINOR.PATCH`
bumps track the API compatibility constraints in [`docs/api.md`](docs/api.md) (breaking
changes to public methods/parameters/return structures bump `MAJOR`). The single
source of truth for the current version is `dragon3_pipelines.__version__`.
See [`CHANGELOG.md`](CHANGELOG.md) (Keep a Changelog format) for what changed in each
release, and [`AGENTS.md`](AGENTS.md) for the release checklist and `scripts/release.sh`.

## Citation

See [`CITATION.cff`](CITATION.cff). A Zenodo DOI is not yet minted for this project;
if that changes, this section and `CITATION.cff` will be updated with the DOI.

## Contributing

Feel free to open issues or submit pull requests!

## License

MIT
