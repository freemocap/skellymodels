"""
Dot-access wrapper for dictionaries.

Enables attribute-style access: `wrapper.some_key` instead of `d["some_key"]`.
Read-only. Raises AttributeError with helpful message listing available keys.
"""


class DotAccessDict:
    """
    Read-only wrapper providing attribute access to a dictionary's values.

    Usage:
        d = {"skull": skull_rb, "pelvis": pelvis_rb}
        wrapper = DotAccessDict(d)
        wrapper.skull      # → skull_rb
        wrapper.pelvis     # → pelvis_rb
        wrapper.nonexistent  # → AttributeError with available keys
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(
            f"'{name}' not found. Available: {sorted(data.keys())}"
        )

    def __dir__(self) -> list[str]:
        """Support tab-completion in IDEs and REPLs."""
        data = object.__getattribute__(self, "_data")
        return sorted(data.keys())

    def __contains__(self, key: str) -> bool:
        data = object.__getattribute__(self, "_data")
        return key in data

    def __iter__(self):
        data = object.__getattribute__(self, "_data")
        return iter(data)

    def __len__(self) -> int:
        data = object.__getattribute__(self, "_data")
        return len(data)

    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        return f"DotAccessDict({sorted(data.keys())})"

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("DotAccessDict is read-only")
