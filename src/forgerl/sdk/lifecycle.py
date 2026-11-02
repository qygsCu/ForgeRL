"""Shared plugin lifecycle contracts."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class PluginContext:
    config: BaseModel


class PluginLifecycle(Protocol):
    def initialize(self, context: PluginContext) -> None:
        ...

    def shutdown(self) -> None:
        ...