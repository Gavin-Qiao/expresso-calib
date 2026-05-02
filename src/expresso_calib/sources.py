from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from .detection import Frame


class CameraSource(Protocol):
    source_id: str

    async def frames(self) -> AsyncIterator[Frame]:
        """Yield decoded frames in the common detector format."""
        yield  # pragma: no cover


@dataclass(frozen=True)
class BrowserUploadCameraSource:
    source_id: str = "browser_upload"


@dataclass(frozen=True)
class MjpegCameraSource:
    url: str
    source_id: str

    async def frames(self) -> AsyncIterator[Frame]:
        raise NotImplementedError(
            "Future source: read PhotonVision/Orange Pi MJPEG streams and yield Frame."
        )
