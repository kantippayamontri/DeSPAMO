import re
from typing import Any


def _is_secret_key(key: object) -> bool:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(key))
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    parts = [part for part in re.split(r"[^a-z0-9]+", name.casefold()) if part]
    return bool(parts) and (
        parts[-1] in {"token", "secret", "password", "credential", "credentials"}
        or parts[-2:] == ["api", "key"]
    )


def reject_secret_keys(value: Any, location: str = "config") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if _is_secret_key(key):
                raise ValueError(f"cannot serialize secret config key: {location}.{key}")
            reject_secret_keys(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_secret_keys(nested, f"{location}[{index}]")
