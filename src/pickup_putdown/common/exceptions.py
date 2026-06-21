"""Shared exception hierarchy and exit-code conventions."""

import sys


class PickupPutdownError(Exception):
    """Base exception for all package errors."""


class ConfigError(PickupPutdownError):
    """Raised when configuration is invalid or cannot be loaded."""


class ValidationError(PickupPutdownError):
    """Raised when data fails schema validation."""


class DataError(PickupPutdownError):
    """Raised when data I/O or processing fails."""


class ExecutionError(PickupPutdownError):
    """Raised when a pipeline step fails."""


EXIT_CODES = {
    PickupPutdownError: 1,
    ConfigError: 2,
    ValidationError: 3,
    DataError: 4,
    ExecutionError: 5,
}


def get_exit_code(exc: Exception) -> int:
    """Return the appropriate exit code for an exception type."""
    for cls in type(exc).__mro__:
        if cls in EXIT_CODES:
            return EXIT_CODES[cls]
    return 1


def exit_with_error(exc: Exception) -> None:
    """Print error message to stderr and exit with appropriate code."""
    import logging

    logging.getLogger(__name__).exception("%s", exc)
    sys.exit(get_exit_code(exc))
