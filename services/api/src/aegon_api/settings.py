"""Validated API configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Scope = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9._-]*(?::[a-z0-9*][a-z0-9*._-]*)*$",
    ),
]


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class TenantKeyGrant(BaseModel):
    """Server-owned tenant and scope assignment for a hashed tenant key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scopes: tuple[Scope, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_scopes(self) -> Self:
        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError("tenant scopes must be unique")
        return self


class ApiSettings(BaseSettings):
    """Settings loaded from ``AEGON_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AEGON_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    environment: Environment = Environment.PRODUCTION
    database_url: SecretStr | None = None
    allowed_origins: tuple[str, ...] = Field(min_length=1)
    tenant_keys: tuple[TenantKeyGrant, ...] = Field(min_length=1)
    enable_demo_bearer_token: bool = False
    demo_bearer_token: SecretStr | None = None
    session_ttl_seconds: int = Field(default=1_800, ge=60, le=86_400)
    max_request_bytes: int = Field(default=1_048_576, ge=1_024, le=10_485_760)
    session_cookie_name: str = Field(
        default="__Host-aegon_session", max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )
    csrf_cookie_name: str = Field(
        default="__Host-aegon_csrf", max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if len(set(self.allowed_origins)) != len(self.allowed_origins):
            raise ValueError("allowed origins must be unique")
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            try:
                _ = parsed.port
            except ValueError as exception:
                raise ValueError("allowed origins must be canonical HTTP origins") from exception
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname is None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or origin != f"{parsed.scheme}://{parsed.netloc}"
            ):
                raise ValueError("allowed origins must be canonical HTTP origins")
        if self.environment is Environment.PRODUCTION:
            if self.database_url is None:
                raise ValueError("database_url is required in production")
            if self.enable_demo_bearer_token:
                raise ValueError("demo bearer authentication cannot be enabled in production")
            if any(urlsplit(origin).scheme != "https" for origin in self.allowed_origins):
                raise ValueError("production origins must use https")
        if self.enable_demo_bearer_token and self.demo_bearer_token is None:
            raise ValueError("demo_bearer_token is required when demo authentication is enabled")
        if len({grant.key_sha256 for grant in self.tenant_keys}) != len(self.tenant_keys):
            raise ValueError("tenant key hashes must be unique")
        if not self.session_cookie_name.startswith("__Host-"):
            raise ValueError("session cookie name must use the __Host- prefix")
        if not self.csrf_cookie_name.startswith("__Host-"):
            raise ValueError("CSRF cookie name must use the __Host- prefix")
        if self.session_cookie_name == self.csrf_cookie_name:
            raise ValueError("session and CSRF cookie names must differ")
        return self
