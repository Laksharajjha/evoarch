"""M/M/c traffic simulation for an :class:`ArchitectureGenome`."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, exp, isfinite, log
from numbers import Real
from random import Random
from typing import Collection, TypedDict

from evoarch.models.genome import ArchitectureGenome, EdgeGene, ServiceGene


class TrafficSimulationError(ValueError):
    """Raised when a genome or workload cannot be simulated meaningfully."""


@dataclass(frozen=True)
class ChaosScenario:
    """A shared, one-replica failure scenario for a population evaluation.

    A service in ``failed_services`` loses one active replica. This models a
    complete pod/node failure while allowing a service with multiple replicas to
    remain available. Applying the same scenario to every candidate keeps each
    NSGA-II generation comparable under identical fault pressure.
    """

    failed_services: frozenset[str]
    failure_ratio: float

    @classmethod
    def sample(
        cls,
        service_names: Collection[str],
        random_source: Random,
    ) -> ChaosScenario:
        """Select a random 10--20% service sample for one fault injection."""
        normalized_names = tuple(sorted(set(service_names)))
        if not normalized_names:
            raise TrafficSimulationError("chaos mode requires at least one service")
        if any(not isinstance(name, str) or not name for name in normalized_names):
            raise TrafficSimulationError("chaos service names must be non-empty strings")

        failure_ratio = random_source.uniform(0.10, 0.20)
        failed_count = min(
            len(normalized_names),
            max(1, ceil(len(normalized_names) * failure_ratio)),
        )
        return cls(
            failed_services=frozenset(random_source.sample(normalized_names, failed_count)),
            failure_ratio=failure_ratio,
        )


class SaturationDetails(TypedDict):
    """Capacity information for a queue whose arrival rate exceeds capacity."""

    arrival_rate_qps: float
    capacity_qps: float
    utilization: float | None
    excess_qps: float


class ServiceLoadDetails(TypedDict):
    """Metrics computed for an individual service in a traffic simulation."""

    arrival_rate_qps: float
    service_rate_per_replica_qps: float
    capacity_qps: float
    utilization: float | None
    erlang_c_wait_probability: float | None
    p99_latency_ms: float | None
    configured_replicas: int
    available_replicas: int
    chaos_failure_injected: bool


class TrafficSimulationResult(TypedDict):
    """The aggregate result returned by :meth:`TrafficSimulator.simulate_load`."""

    total_p99_latency_ms: float | None
    total_cost_hourly: float
    queue_saturation: dict[str, SaturationDetails]
    service_metrics: dict[str, ServiceLoadDetails]
    chaos_mode: bool
    chaos_failed_services: tuple[str, ...]


class TrafficSimulator:
    """Simulate service load, M/M/c response times, and resource cost.

    A root service receives ``baseline_qps`` from outside the topology. Every
    outgoing edge propagates the source service's full arrival rate, modelling a
    synchronous call on each declared dependency. The resulting graph must be a
    directed acyclic graph so traffic can be propagated deterministically.
    """

    CPU_HOURLY_PRICE = 0.04
    GIB_MEMORY_HOURLY_PRICE = 0.01

    def __init__(self, base_service_rate_per_cpu_qps: float = 100.0) -> None:
        """Create a simulator with a calibrated per-CPU service rate in QPS."""
        self._base_service_rate_per_cpu_qps = self._validate_positive_finite(
            base_service_rate_per_cpu_qps,
            "base_service_rate_per_cpu_qps",
        )

    def simulate_load(
        self,
        genome: ArchitectureGenome,
        baseline_qps: float,
        *,
        chaos_mode: bool = False,
        chaos_scenario: ChaosScenario | None = None,
    ) -> TrafficSimulationResult:
        """Simulate ``genome`` at the supplied external request rate.

        A saturated M/M/c queue has no finite steady-state P99, which is
        represented by ``None`` in both the service and aggregate latency fields.
        Its capacity details remain available in ``queue_saturation``.
        """
        if not isinstance(genome, ArchitectureGenome):
            raise TrafficSimulationError("genome must be an ArchitectureGenome")
        if not isinstance(chaos_mode, bool):
            raise TrafficSimulationError("chaos_mode must be a boolean")
        if chaos_scenario is not None and not isinstance(chaos_scenario, ChaosScenario):
            raise TrafficSimulationError("chaos_scenario must be a ChaosScenario")
        if chaos_scenario is not None and not chaos_mode:
            raise TrafficSimulationError("chaos_scenario requires chaos_mode=True")

        external_qps = self._validate_positive_finite(baseline_qps, "baseline_qps")
        outgoing_edges, incoming_counts, topological_order = self._build_flow_graph(
            genome
        )
        arrival_rates = self._propagate_arrival_rates(
            outgoing_edges,
            incoming_counts,
            topological_order,
            external_qps,
        )
        active_scenario = chaos_scenario
        if chaos_mode and active_scenario is None:
            active_scenario = ChaosScenario.sample(genome.services.keys(), Random())
        failed_services = (
            active_scenario.failed_services if active_scenario is not None else frozenset()
        )
        unknown_failed_services = failed_services.difference(genome.services)
        if unknown_failed_services:
            unknown_names = ", ".join(sorted(unknown_failed_services))
            raise TrafficSimulationError(
                f"chaos scenario contains unknown services: {unknown_names}"
            )

        service_metrics: dict[str, ServiceLoadDetails] = {}
        queue_saturation: dict[str, SaturationDetails] = {}
        per_service_p99_ms: dict[str, float | None] = {}

        for service_name in topological_order:
            service = genome.services[service_name]
            chaos_failure_injected = service_name in failed_services
            metrics = self._simulate_service(
                service,
                arrival_rates[service_name],
                available_replicas=(
                    max(0, service.replicas - 1)
                    if chaos_failure_injected
                    else service.replicas
                ),
                chaos_failure_injected=chaos_failure_injected,
            )
            service_metrics[service_name] = metrics
            per_service_p99_ms[service_name] = metrics["p99_latency_ms"]

            if metrics["p99_latency_ms"] is None:
                queue_saturation[service_name] = {
                    "arrival_rate_qps": metrics["arrival_rate_qps"],
                    "capacity_qps": metrics["capacity_qps"],
                    "utilization": metrics["utilization"],
                    "excess_qps": (
                        metrics["arrival_rate_qps"] - metrics["capacity_qps"]
                    ),
                }

        total_p99_latency_ms = self._critical_path_p99(
            topological_order,
            outgoing_edges,
            per_service_p99_ms,
        )
        return {
            "total_p99_latency_ms": total_p99_latency_ms,
            "total_cost_hourly": self._compute_hourly_cost(genome),
            "queue_saturation": queue_saturation,
            "service_metrics": service_metrics,
            "chaos_mode": chaos_mode,
            "chaos_failed_services": tuple(sorted(failed_services)),
        }

    def _simulate_service(
        self,
        service: ServiceGene,
        arrival_rate_qps: float,
        *,
        available_replicas: int,
        chaos_failure_injected: bool,
    ) -> ServiceLoadDetails:
        service_rate_qps = (
            service.cpu_limit * self._base_service_rate_per_cpu_qps
        )
        capacity_qps = available_replicas * service_rate_qps
        if available_replicas == 0:
            return {
                "arrival_rate_qps": arrival_rate_qps,
                "service_rate_per_replica_qps": service_rate_qps,
                "capacity_qps": 0.0,
                "utilization": None,
                "erlang_c_wait_probability": None,
                "p99_latency_ms": None,
                "configured_replicas": service.replicas,
                "available_replicas": 0,
                "chaos_failure_injected": chaos_failure_injected,
            }
        utilization = arrival_rate_qps / capacity_qps

        if utilization >= 1.0:
            return {
                "arrival_rate_qps": arrival_rate_qps,
                "service_rate_per_replica_qps": service_rate_qps,
                "capacity_qps": capacity_qps,
                "utilization": utilization,
                "erlang_c_wait_probability": None,
                "p99_latency_ms": None,
                "configured_replicas": service.replicas,
                "available_replicas": available_replicas,
                "chaos_failure_injected": chaos_failure_injected,
            }

        wait_probability = self._erlang_c(
            arrival_rate_qps,
            service_rate_qps,
            service.replicas,
        )
        return {
            "arrival_rate_qps": arrival_rate_qps,
            "service_rate_per_replica_qps": service_rate_qps,
            "capacity_qps": capacity_qps,
            "utilization": utilization,
            "erlang_c_wait_probability": wait_probability,
            "p99_latency_ms": 1000.0
            * self._response_time_quantile(
                arrival_rate_qps,
                service_rate_qps,
                service.replicas,
                wait_probability,
                0.99,
            ),
            "configured_replicas": service.replicas,
            "available_replicas": available_replicas,
            "chaos_failure_injected": chaos_failure_injected,
        }

    @staticmethod
    def _erlang_c(arrival_rate_qps: float, service_rate_qps: float, servers: int) -> float:
        """Return the exact Erlang C probability that an arrival must wait."""
        offered_load = arrival_rate_qps / service_rate_qps
        utilization = offered_load / servers
        if utilization >= 1.0:
            return 1.0

        series_sum = 1.0
        term = 1.0
        for server_count in range(1, servers):
            term *= offered_load / server_count
            series_sum += term

        final_term = term * offered_load / servers
        queue_term = final_term / (1.0 - utilization)
        return queue_term / (series_sum + queue_term)

    @staticmethod
    def _response_time_quantile(
        arrival_rate_qps: float,
        service_rate_qps: float,
        servers: int,
        wait_probability: float,
        quantile: float,
    ) -> float:
        """Solve the exact M/M/c response-time quantile by bounded bisection."""
        wait_rate = servers * service_rate_qps - arrival_rate_qps

        def response_cdf(time_seconds: float) -> float:
            no_wait_cdf = 1.0 - exp(-service_rate_qps * time_seconds)
            if wait_probability == 0.0:
                return no_wait_cdf

            if abs(wait_rate - service_rate_qps) < 1e-12:
                wait_cdf = 1.0 - exp(-service_rate_qps * time_seconds) * (
                    1.0 + service_rate_qps * time_seconds
                )
            else:
                survival_probability = (
                    wait_rate * exp(-service_rate_qps * time_seconds)
                    - service_rate_qps * exp(-wait_rate * time_seconds)
                ) / (wait_rate - service_rate_qps)
                wait_cdf = 1.0 - survival_probability

            return (1.0 - wait_probability) * no_wait_cdf + wait_probability * wait_cdf

        upper_bound = log(1.0 / (1.0 - quantile)) / min(
            service_rate_qps,
            wait_rate,
        )
        while response_cdf(upper_bound) < quantile:
            upper_bound *= 2.0

        lower_bound = 0.0
        for _ in range(80):
            midpoint = (lower_bound + upper_bound) / 2.0
            if response_cdf(midpoint) < quantile:
                lower_bound = midpoint
            else:
                upper_bound = midpoint
        return upper_bound

    @staticmethod
    def _build_flow_graph(
        genome: ArchitectureGenome,
    ) -> tuple[dict[str, list[EdgeGene]], dict[str, int], list[str]]:
        """Return adjacency data and a stable topological order for the genome."""
        outgoing_edges = {service_name: [] for service_name in genome.services}
        incoming_counts = {service_name: 0 for service_name in genome.services}

        for edge in genome.edges:
            outgoing_edges[edge.source].append(edge)
            incoming_counts[edge.target] += 1

        remaining_incoming = incoming_counts.copy()
        ready = deque(
            service_name
            for service_name in genome.services
            if remaining_incoming[service_name] == 0
        )
        topological_order: list[str] = []

        while ready:
            service_name = ready.popleft()
            topological_order.append(service_name)
            for edge in outgoing_edges[service_name]:
                remaining_incoming[edge.target] -= 1
                if remaining_incoming[edge.target] == 0:
                    ready.append(edge.target)

        if len(topological_order) != len(genome.services):
            raise TrafficSimulationError(
                "traffic simulation requires an acyclic service dependency graph"
            )

        return outgoing_edges, incoming_counts, topological_order

    @staticmethod
    def _propagate_arrival_rates(
        outgoing_edges: dict[str, list[EdgeGene]],
        incoming_counts: dict[str, int],
        topological_order: list[str],
        baseline_qps: float,
    ) -> dict[str, float]:
        """Propagate external and upstream requests through the topology."""
        arrival_rates = {
            service_name: baseline_qps if incoming_counts[service_name] == 0 else 0.0
            for service_name in topological_order
        }
        for service_name in topological_order:
            for edge in outgoing_edges[service_name]:
                arrival_rates[edge.target] += arrival_rates[service_name]
        return arrival_rates

    @staticmethod
    def _critical_path_p99(
        topological_order: list[str],
        outgoing_edges: dict[str, list[EdgeGene]],
        per_service_p99_ms: dict[str, float | None],
    ) -> float | None:
        """Calculate the slowest root-to-service path including edge latency."""
        if any(latency is None for latency in per_service_p99_ms.values()):
            return None

        cumulative_latency = {
            service_name: 0.0 for service_name in topological_order
        }
        for service_name in topological_order:
            service_latency = per_service_p99_ms[service_name]
            if service_latency is None:
                return None
            cumulative_latency[service_name] += service_latency
            for edge in outgoing_edges[service_name]:
                cumulative_latency[edge.target] = max(
                    cumulative_latency[edge.target],
                    cumulative_latency[service_name] + edge.base_latency_ms,
                )

        return max(cumulative_latency.values())

    @classmethod
    def _compute_hourly_cost(cls, genome: ArchitectureGenome) -> float:
        """Compute the standard hourly price for every configured replica."""
        return sum(
            service.replicas
            * (
                service.cpu_limit * cls.CPU_HOURLY_PRICE
                + (service.mem_limit_mb / 1024.0) * cls.GIB_MEMORY_HOURLY_PRICE
            )
            for service in genome.services.values()
        )

    @staticmethod
    def _validate_positive_finite(value: float, field_name: str) -> float:
        """Validate numeric inputs before they enter the queueing equations."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TrafficSimulationError(f"{field_name} must be a real number")
        numeric_value = float(value)
        if not isfinite(numeric_value) or numeric_value <= 0.0:
            raise TrafficSimulationError(
                f"{field_name} must be a finite value greater than zero"
            )
        return numeric_value


def simulate_load(
    genome: ArchitectureGenome,
    baseline_qps: float,
    *,
    chaos_mode: bool = False,
    chaos_scenario: ChaosScenario | None = None,
) -> TrafficSimulationResult:
    """Simulate load with EvoArch's standard traffic calibration."""
    return TrafficSimulator().simulate_load(
        genome,
        baseline_qps,
        chaos_mode=chaos_mode,
        chaos_scenario=chaos_scenario,
    )
