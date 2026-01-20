"""Status codes and error handling for ndtensors-rs."""

from enum import IntEnum


class StatusCode(IntEnum):
    """Status codes returned by the C API."""

    SUCCESS = 0
    INVALID_ARGUMENT = -1
    SHAPE_MISMATCH = -2
    INDEX_OUT_OF_BOUNDS = -3
    INTERNAL_ERROR = -4
    INVALID_PERMUTATION = -5


class NDTensorsError(Exception):
    """Base exception for ndtensors errors."""

    pass


def check_status(status: int, context: str = "") -> None:
    """Raise exception if status indicates error.

    Args:
        status: Status code from C API
        context: Description of the operation for error message
    """
    if status != StatusCode.SUCCESS:
        msg = f"{context}: {StatusCode(status).name}" if context else StatusCode(status).name
        raise NDTensorsError(msg)
