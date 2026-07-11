# nbody-pipeline Scripts

This directory contains shell scripts for the nbody_pipeline package.

## Scripts

### nbody_jpg_to_movie.sh
Main script for creating movies from JPG snapshots.

```bash
# Create movies from all preset plot patterns
bash nbody_jpg_to_movie.sh

# Create movies for one or more selected/custom plot patterns
bash nbody_jpg_to_movie.sh create _CMD.jpg _custom_suffix.jpg

# Show top-level help or preset plot patterns
bash nbody_jpg_to_movie.sh --help
bash nbody_jpg_to_movie.sh create help
```

## Backward Compatibility

For backward compatibility, these scripts are also available in the repository root directory.
