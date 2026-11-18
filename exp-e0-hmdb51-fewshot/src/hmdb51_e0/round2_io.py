"""Output boundaries for the additive second-round workflow."""
from pathlib import Path


def safe_output(output: Path, paths: dict, protected=()) -> Path:
    output = Path(output).resolve()
    for key in ('dataset_root', 'feature_cache_root', 'weights_file', 'annotations_root'):
        path = paths.get(key)
        if path is not None:
            path = Path(path).resolve()
            if output == path or output.is_relative_to(path) or path.is_relative_to(output):
                raise ValueError(f'output overlaps protected input: {key}')
    for path in protected:
        if output.is_relative_to(Path(path).resolve()):
            raise ValueError('output is inside a protected first-round artifact directory')
    return output
