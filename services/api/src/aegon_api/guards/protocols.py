"""Dependency-injection protocols for safety controls."""

from typing import Protocol

from aegon_api.guards.models import GuardResult, ScreeningDecision


class Guard[PayloadT, ResultT](Protocol):
    """Evaluate one bounded payload without invoking agent tools."""

    async def evaluate(self, payload: PayloadT) -> GuardResult[ResultT]: ...


class ManagedScreeningService(Protocol):
    """Optional adapter for a managed content-screening service."""

    async def screen(self, text: str) -> ScreeningDecision: ...
