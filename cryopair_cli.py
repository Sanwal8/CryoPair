"""Load optional dependencies for the slide prediction and adapter training commands."""
import sys
from importlib import import_module


def launch(module, extra):
    try:
        program = import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "pandas"}:
            raise
        print(f'error: missing {exc.name}. From the repository, run python -m pip install ".[{extra}]"', file=sys.stderr)
        return 2
    return program.main()


def predict():
    return launch("cryopair_predict", "slides")


def train():
    return launch("cryopair_train", "train")
