"""
Push notifications via ntfy (https://ntfy.sh).

Fires off a POST in a daemon thread when a swing is detected so the
detection pipeline never waits on the network. Configured under the
top-level `notifications:` block in config.yaml.
"""

import logging
import threading
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(
        self,
        enabled: bool = False,
        server: str = "https://ntfy.sh",
        topic: str | None = None,
        priority: int = 4,
        tags: str = "golf",
        click_url: str | None = None,
        timeout: float = 5.0,
    ):
        self.enabled = bool(enabled and topic)
        self.server = server.rstrip("/")
        self.topic = topic
        self.priority = str(priority)
        self.tags = tags
        self.click_url = click_url
        self.timeout = timeout

        if enabled and not topic:
            logger.warning("ntfy enabled but no topic configured — disabling")

    def notify_swing(self, swing_id: str, impact_peak: float | None = None):
        """Fire-and-forget swing notification. Returns immediately."""
        if not self.enabled:
            return

        if impact_peak is not None:
            title = "⛳ Swing detected"
            body = f"Impact heard (peak {impact_peak:.2f})"
        else:
            title = "⛳ Swing detected"
            body = "Tap to review"

        threading.Thread(
            target=self._post,
            args=(title, body),
            daemon=True,
            name=f"ntfy-{swing_id[:8]}",
        ).start()

    def _post(self, title: str, body: str):
        url = f"{self.server}/{self.topic}"
        headers = {
            "Title": title,
            "Priority": self.priority,
            "Tags": self.tags,
        }
        if self.click_url:
            headers["Click"] = self.click_url

        req = urllib.request.Request(
            url, data=body.encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    logger.warning(f"ntfy returned {resp.status}")
        except urllib.error.URLError as e:
            logger.warning(f"ntfy post failed: {e}")
        except Exception as e:
            logger.warning(f"ntfy post failed: {e}")
