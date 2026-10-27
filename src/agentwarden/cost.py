"""Spend ledger and guard (stub)."""


class SpendRefused(Exception):
    """Raised when a reservation is refused."""


class Ledger:
    """Append-only spend ledger (stub)."""

    def __init__(self, path, global_cap_usd: float = 100.0) -> None:
        raise NotImplementedError
