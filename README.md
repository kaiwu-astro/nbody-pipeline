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
config = ConfigManager()
config.processes_count = 20

# Use visualizers
visualizer = SingleStarVisualizer(config)

# Read the cached initial total mass from lagr.7
initial_mass_msun = InitialTotalMassAnalyzer(config).get_initial_total_mass_msun("20sb")
```

## Configuration

### Using YAML Configuration

Create a custom `config.yaml`:

```yaml
paths:
  simulations:
    my_sim: "/path/to/simulation"
  plot_dir: "/path/to/plots"

processing:
  processes_count: 40
  skip_existing_plot: true

hdf5:
  file_selection:
    wait_age_hour: 24
    sample_every_nb_time: 1.0
    exclude_bad_dirname: true
  table_cache:
    use_hdf5_cache: true
  scan:
    parallel: false
    incremental_from_cache_tail: true

galactic_orbit:
  enabled: true
  time_color_max_myr: 500.0
```

Load it in your code:

```python
from dragon3_pipelines.config import load_config
config = load_config("config.yaml")
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
- **Comprehensive Testing**: 87+ unit tests covering all modules

## Usage Examples

### Analyze HDF5 Files

```python
from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
processor = HDF5FileProcessor(config)
df_dict = processor.read_file(hdf5_path="/path/to/file.h5part", simu_name="my_sim")
# df_dict contains 'scalars', 'singles', 'binaries', 'mergers' DataFrames
```

### Track Particles

```python
from dragon3_pipelines.analysis import ParticleTracker
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
tracker = ParticleTracker(config)
particle_history = tracker.update_one_particle_history_df(simu_name="my_sim", particle_name=12345)
```

### Extract Binaries by Stellar Type

```python
from dragon3_pipelines.analysis import BinaryStellarTypeExtractor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
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

### Create Visualizations

```python
from dragon3_pipelines.visualization import BinaryStarVisualizer
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
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
- `--debug`: Enable debug logging

The installed `dragon3-plot` script accepts the same arguments as `python -m dragon3_pipelines`.

### Purge Generated Plots

```bash
# List supported purge targets
python -m dragon3_pipelines purge --list-targets

# Preview matching files before deleting
python -m dragon3_pipelines purge single.create_position_plot_jpg --simu sim_a --dry-run

# Delete matching files without an interactive confirmation
python -m dragon3_pipelines purge single.create_position_plot_jpg --simu sim_a --yes
```

## Contributing

Feel free to open issues or submit pull requests!

## License

MIT
