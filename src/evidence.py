"""Evidence helper: build numbered evidence items for one subagent."""

from __future__ import annotations

from .models import EvidenceItem


class EvidenceBuilder:
    """Collects evidence for one subagent and hands out stable ids such as MKT-1, MKT-2."""

    def __init__(self, prefix: str, source: str, as_of: str | None = None, retrieved_at: str | None = None):
        self.prefix = prefix
        self.defaults = {"source": source, "as_of": as_of}
        if retrieved_at:
            self.defaults["retrieved_at"] = retrieved_at
        self.items: list[EvidenceItem] = []

    def add(self, claim: str, value=None, unit: str | None = None, **overrides) -> str:
        """Record one fact and return its id so a finding can cite it."""
        item = EvidenceItem(
            id=f"{self.prefix}-{len(self.items) + 1}", claim=claim, value=value, unit=unit,
            **{**self.defaults, **overrides},
        )
        self.items.append(item)
        return item.id
