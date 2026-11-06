"""Trust labels attached to every piece of text an agent sees."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Trust(IntEnum):
    UNTRUSTED = 0
    TRUSTED = 1


@dataclass(frozen=True)
class Source:
    """Where a piece of text came from.

    ``kind`` is ``"user"``, ``"system"`` or ``"tool"``; ``name`` identifies the
    tool for tool outputs. ``secret`` marks confidential data (a passport
    number, an account balance) that must not flow into outgoing free text;
    it is independent of ``trust``, which is about who can write the data.
    """

    kind: str
    name: str = ""
    trust: Trust = Trust.UNTRUSTED
    secret: bool = False

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}" if self.name else self.kind


USER = Source("user", trust=Trust.TRUSTED)
SYSTEM = Source("system", trust=Trust.TRUSTED)
