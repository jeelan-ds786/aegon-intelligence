"""Public API error contracts and exceptions."""

from __future__ import annotations

from typing import Annotated

from aegon_common.models import ContractModel, ErrorCode
from pydantic import Field


class ErrorDetail(ContractModel):
    code: ErrorCode
    message: Annotated[str, Field(min_length=1, max_length=2_000)]
    retryable: bool = False
    request_id: Annotated[str, Field(min_length=1, max_length=128)]


class ErrorEnvelope(ContractModel):
    error: ErrorDetail


class ApiError(Exception):
    """A failure safe to expose through the public error envelope."""

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
