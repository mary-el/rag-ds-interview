import os

import yaml

_config = None


def load_config(path=None) -> dict:
    global _config
    if _config is None:
        if path is None:
            raise ValueError("Config is not initialized")
        with open(path, "r") as f:
            raw = f.read()
        raw = os.path.expandvars(raw)  # loading env variables
        _config = yaml.safe_load(raw)
    return _config
