"""Pydantic models that define an EvoArch microservice topology genome."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RoutingAlgorithm = Literal["round_robin", "least_connections", "random"]


class ServiceGene(BaseModel):
    """Resource and routing characteristics for one microservice."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    service_name: str = Field(min_length=1, max_length=255)
    replicas: int = Field(ge=1, le=20)
    cpu_limit: float = Field(ge=0.1, le=4.0)
    mem_limit_mb: int = Field(ge=128, le=8192)
    routing_algorithm: RoutingAlgorithm

    @field_validator("service_name")
    @classmethod
    def service_name_must_not_be_blank(cls, value: str) -> str:
        """Reject service names composed only of whitespace."""
        if not value:
            raise ValueError("service_name must not be blank")
        return value


class EdgeGene(BaseModel):
    """A directed network path between two services."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    source: str = Field(min_length=1, max_length=255)
    target: str = Field(min_length=1, max_length=255)
    base_latency_ms: float = Field(ge=0.0)

    @field_validator("source", "target")
    @classmethod
    def endpoint_must_not_be_blank(cls, value: str) -> str:
        """Reject edge endpoints composed only of whitespace."""
        if not value:
            raise ValueError("edge endpoint must not be blank")
        return value


class ArchitectureGenome(BaseModel):
    """A validated directed microservice topology and its service genes.

    Parallel edges are permitted because they can represent distinct call paths.
    Edges must reference services in ``services``; cycle validation is deferred to
    the traffic simulator, which requires an acyclic flow graph.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )

    services: dict[str, ServiceGene] = Field(min_length=1)
    edges: list[EdgeGene] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_topology_references(self) -> ArchitectureGenome:
        """Keep service-map keys and all edge references internally consistent."""
        service_names = set(self.services)

        for service_name, service in self.services.items():
            if not service_name.strip():
                raise ValueError("services cannot contain a blank service name")
            if service_name != service.service_name:
                raise ValueError(
                    "service mapping key must match ServiceGene.service_name: "
                    f"{service_name!r} != {service.service_name!r}"
                )

        for edge in self.edges:
            if edge.source not in service_names:
                raise ValueError(
                    f"edge source {edge.source!r} is not defined in services"
                )
            if edge.target not in service_names:
                raise ValueError(
                    f"edge target {edge.target!r} is not defined in services"
                )

        return self
