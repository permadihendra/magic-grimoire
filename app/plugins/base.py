from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class BotContext:
    chat_id: int
    user_id: int
    username: str | None
    message_text: str
    is_group: bool
    raw_update: object  # python-telegram-bot Update object
    thinking_msg_id: int | None = None  # Telegram message_id of the "thinking..." message


# Return type: reply string, None (silent), or DispatchResult(reply, processing_metadata)
PluginResult = Union[str, None, "DispatchResult"]


class Plugin(ABC):
    name: str  # unique snake_case identifier
    commands: list[str]  # telegram slash commands owned by this plugin
    description: str  # shown in /help output

    @abstractmethod
    async def handle(self, ctx: BotContext) -> PluginResult:
        """Return reply string, None (silent), or DispatchResult(reply, processing_metadata)."""
        ...

    async def on_load(self) -> None:
        """Called once at startup. Set up DB tables, schedules, etc."""

    async def on_unload(self) -> None:
        """Called at shutdown."""


class PluginRegistry:
    _instance: "PluginRegistry | None" = None
    _plugins: dict[str, "Plugin"] = {}

    @classmethod
    def get(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, plugin: "Plugin") -> None:
        self._plugins[plugin.name] = plugin

    def get_plugin(self, name: str) -> "Plugin | None":
        """Get a plugin by its name."""
        return self._plugins.get(name)

    def resolve(self, command: str) -> "Plugin | None":
        for p in self._plugins.values():
            if command in p.commands:
                return p
        return None

    def all(self) -> list["Plugin"]:
        return list(self._plugins.values())


class DispatchResult:
    """Result from dispatch: the reply text plus optional processing metadata.

    The `reply` attribute must always be a true string — never None, never
    wrapped in a descriptor. Empty __slots__ caused bool() to return False
    for non-empty strings, breaking all non-/index commands.
    """
    __slots__ = ("reply", "processing")

    def __init__(self, reply: str | None, processing: dict[str, Any] | None = None):
        self.reply = reply
        self.processing = processing

    def __repr__(self) -> str:
        r = self.reply
        s = r[:50] if r else None
        return "DispatchResult(reply=%r)" % s