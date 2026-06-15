"""Notifications. Currently Telegram, with a no-op fallback.

Uses only the standard library (urllib) so it adds no dependency. A failed
notification must never crash the trading loop, so all errors are swallowed and
logged.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

log = logging.getLogger("notify")


class Notifier:
    """Base notifier: does nothing. Used when alerts are not configured."""

    def send(self, text: str) -> None:  # pragma: no cover - trivial
        pass


class TelegramNotifier(Notifier):
    def __init__(self, token: str, chat_id: str, timeout: float = 10.0):
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout

    def send(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        ).encode()
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except Exception as exc:  # never let a notification break trading
            log.warning("Telegram notify failed: %s", exc)


def make_notifier(token: str, chat_id: str) -> Notifier:
    if token and chat_id:
        return TelegramNotifier(token, chat_id)
    return Notifier()
