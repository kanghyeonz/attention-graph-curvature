import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def versions():
    result = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'transformers', 'datasets'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result
