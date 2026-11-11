"""Production command-line entrypoint for the Aegon API."""

import os

import uvicorn
from pydantic import BaseModel, ConfigDict, Field


class LauncherSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    port: int = Field(ge=1, le=65_535)


def run() -> None:
    """Run the API factory with Cloud Run-compatible defaults."""

    settings = LauncherSettings.model_validate({"port": os.environ.get("PORT", "8080")})
    uvicorn.run(
        "aegon_api:create_app",
        factory=True,
        host="0.0.0.0",
        port=settings.port,
        proxy_headers=True,
    )
