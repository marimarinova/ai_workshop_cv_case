"""Tests for the exception hierarchy and exit codes."""


from pickup_putdown.common.exceptions import (
    ConfigError,
    DataError,
    ExecutionError,
    PickupPutdownError,
    ValidationError,
    get_exit_code,
)


class TestExceptionHierarchy:
    def test_base_exception_exists(self):
        assert issubclass(PickupPutdownError, Exception)

    def test_config_error_is_subclass(self):
        assert issubclass(ConfigError, PickupPutdownError)

    def test_validation_error_is_subclass(self):
        assert issubclass(ValidationError, PickupPutdownError)

    def test_data_error_is_subclass(self):
        assert issubclass(DataError, PickupPutdownError)

    def test_execution_error_is_subclass(self):
        assert issubclass(ExecutionError, PickupPutdownError)


class TestExitCodes:
    def test_pickup_putdown_error_returns_1(self):
        assert get_exit_code(PickupPutdownError()) == 1

    def test_config_error_returns_2(self):
        assert get_exit_code(ConfigError()) == 2

    def test_validation_error_returns_3(self):
        assert get_exit_code(ValidationError()) == 3

    def test_data_error_returns_4(self):
        assert get_exit_code(DataError()) == 4

    def test_execution_error_returns_5(self):
        assert get_exit_code(ExecutionError()) == 5

    def test_unknown_exception_returns_1(self):
        assert get_exit_code(ValueError("unexpected")) == 1

    def test_subclass_gets_parent_code(self):
        class MyConfigError(ConfigError):
            pass
        assert get_exit_code(MyConfigError()) == 2
