from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    yaml = None
    _YAML_ERROR = exc
else:
    _YAML_ERROR = None


def load_config(path: str | Path) -> dict[str, Any]:
    if _YAML_ERROR is not None:
        raise ModuleNotFoundError("inhouse inference requires PyYAML") from _YAML_ERROR
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
