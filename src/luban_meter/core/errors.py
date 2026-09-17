"""Framework-level error types."""


class BenchmarkToolkitError(Exception):
    """Base error for LuBan-Meter."""


class ConfigurationError(BenchmarkToolkitError):
    """A configuration file or command-line value is invalid."""


class DuplicateModuleError(BenchmarkToolkitError):
    def __init__(self, name: str) -> None:
        super().__init__(f"module already registered: {name}")


class UnknownModuleError(BenchmarkToolkitError):
    def __init__(self, name: str) -> None:
        super().__init__(f"unknown module: {name}")


class UnknownScenarioError(BenchmarkToolkitError):
    def __init__(self, category_name: str, scenario_name: str) -> None:
        super().__init__(
            f"unknown benchmark for {category_name}: {scenario_name}"
        )


class ExecutionError(BenchmarkToolkitError):
    """A local Benchmark command could not be executed."""


class ResultProcessingError(BenchmarkToolkitError):
    """A Benchmark tool result could not be validated or processed."""
