import os

import yaml

_config = None


def load_config(path="configs/config.yaml") -> dict:
    global _config
    if _config is None:
        with open(path, "r") as f:
            raw = f.read()
        raw = os.path.expandvars(raw)  # loading env variables
        _config = yaml.safe_load(raw)
    return _config
