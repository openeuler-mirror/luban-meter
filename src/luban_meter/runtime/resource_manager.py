"""Parent-owned resource lifetimes, selected by capability declaration."""

from abc import ABC, abstractmethod

from luban_meter.core.registry import Registry
from luban_meter.utils.docker_sandbox import (
    DockerSandboxConfig,
    DockerSandboxScope,
)


class Resource(ABC):
    @abstractmethod
    def open(self, directory, parameters):
        """Return a handle with environment() and cleanup()."""


resources = Registry("resource", Resource)


@resources.register("docker_sandbox")
class DockerSandboxResource(Resource):
    def open(self, directory, parameters):
        return DockerSandboxScope(
            directory / "sandbox-registry",
            DockerSandboxConfig(
                docker_host=parameters["docker_host"],
                image=parameters["sandbox_image"],
                docker_binary=parameters["docker_binary"],
            ),
        )
