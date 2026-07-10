# Dragon3 Pipelines Documentation

## Overview

Dragon3 Pipelines is a modular Python package for analyzing and visualizing N-body simulation data from Dragon3 simulations. It provides a clean, type-annotated API for working with HDF5 files, tracking particles, and creating publication-quality visualizations.

## Installation

```bash
# Clone the repository
git clone https://github.com/kaiwu-astro/dragon3_pipeline.git
cd dragon3_pipeline

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

### Command Line Usage

```bash
# Show available commands and options
python -m dragon3_pipelines --help

# Run analysis pipeline
python -m dragon3_pipelines

# Resume from existing plots
python -m dragon3_pipelines --skip-until=last

# Show purge command help
python -m dragon3_pipelines help purge

# Generate movies
bash dragon3_jpg_to_movie.sh
```

### Python API

```python
from dragon3_pipelines import SimulationPlotter
from dragon3_pipelines.config import ConfigManager
from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.analysis import ParticleTracker
from dragon3_pipelines.visualization import SingleStarVisualizer

# Load configuration
config = ConfigManager()

# Process HDF5 files
processor = HDF5FileProcessor(config, "my_simulation")
df_singles, df_binaries = processor.get_hdf5_dataframes()

# Track particles
tracker = ParticleTracker(config, "my_simulation")
history = tracker.update_one_particle_history_df(simu_name='my_sim', particle_id=12345)

# Create visualizations
viz = SingleStarVisualizer(processor, config)
viz.create_mass_vs_distance_plot(df_singles, time=100.0)
```

## Modules

### config
Configuration management with YAML support. Define simulation paths, processing options, and plotting parameters.

### io
Data I/O operations for HDF5 files, Lagrangian radii files, collision data, and text-based outputs.

### analysis
Particle tracking through simulation snapshots and physics calculations (gravitational wave timescales, etc.).

### visualization
Create publication-quality plots and figures from simulation data.

### utils
Utility functions for serialization, shell commands, logging, and color conversion.

## Features

- ✅ **Type-annotated**: Full type hints for better IDE support and type checking
- ✅ **Modular**: Clean separation of concerns with well-defined interfaces
- ✅ **Well-tested**: Comprehensive test suite with 87+ tests
- ✅ **Configurable**: YAML-based configuration with sensible defaults
- ✅ **Parallel processing**: Multi-process support for large datasets

## Documentation

- [API Reference](api.md) - Detailed API documentation
- [Unified Analysis Architecture](analysis_architecture.md) - Data layering, scan task taxonomy, and the Parquet/DuckDB analysis layer
- [Configuration Guide](../README.md#configuration) - How to configure the package
- [Examples](../README.md#usage-examples) - Usage examples

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT
