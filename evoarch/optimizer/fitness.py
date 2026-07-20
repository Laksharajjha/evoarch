"""NSGA-II fitness evaluation for EvoArch architecture genomes."""

from __future__ import annotations

from math import inf, isfinite
from numbers import Real
from random import Random
from typing import Sequence, TypedDict

from evoarch.models.genome import ArchitectureGenome
from evoarch.simulation.traffic import (
    ChaosScenario,
    ServiceLoadDetails,
    SaturationDetails,
    TrafficSimulationResult,
    TrafficSimulator,
)

DEFAULT_BASELINE_QPS = 100.0
DEFAULT_LATENCY_WEIGHT = 0.7
DEFAULT_COST_WEIGHT = 0.3
SATURATION_PENALTY = 10_000.0
SIMULATION_FAILURE_PENALTY = 100_000.0


class FitnessRecord(TypedDict):
    """Fitness, simulation, and diversity data for one population member."""

    genome_index: int
    total_p99_latency_ms: float | None
    total_cost_hourly: float
    queue_saturation: dict[str, SaturationDetails]
    service_metrics: dict[str, ServiceLoadDetails]
    chaos_mode: bool
    chaos_failed_services: tuple[str, ...]
    simulation_error: str | None
    structural_penalties: float
    front_rank: int
    crowding_distance: float
    composite_fitness: float


def calculate_pareto_fitness(
    genomes: list[ArchitectureGenome],
    simulator: TrafficSimulator,
    *,
    baseline_qps: float = DEFAULT_BASELINE_QPS,
    latency_weight: float = DEFAULT_LATENCY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    chaos_mode: bool = False,
    chaos_scenario: ChaosScenario | None = None,
) -> list[FitnessRecord]:
    """Evaluate genomes and rank them with NSGA-II minimization objectives.

    Latency and hourly resource cost are minimized. A saturated queue or a failed
    simulation has unbounded latency, zero scalar fitness, and remains in the
    population so the evolutionary engine can recover through mutation.
    """
    _validate_inputs(genomes, simulator, baseline_qps)
    normalized_latency_weight, normalized_cost_weight = _validate_objective_weights(
        latency_weight,
        cost_weight,
    )
    if not isinstance(chaos_mode, bool):
        raise ValueError("chaos_mode must be a boolean")
    if chaos_scenario is not None and not isinstance(chaos_scenario, ChaosScenario):
        raise TypeError("chaos_scenario must be a ChaosScenario")
    if chaos_scenario is not None and not chaos_mode:
        raise ValueError("chaos_scenario requires chaos_mode=True")

    shared_chaos_scenario = chaos_scenario
    if chaos_mode and shared_chaos_scenario is None and genomes:
        shared_chaos_scenario = ChaosScenario.sample(
            genomes[0].services.keys(),
            Random(),
        )

    simulation_results: list[TrafficSimulationResult | None] = []
    simulation_errors: list[str | None] = []
    for genome in genomes:
        try:
            simulation_results.append(
                simulator.simulate_load(
                    genome,
                    baseline_qps,
                    chaos_mode=chaos_mode,
                    chaos_scenario=shared_chaos_scenario,
                )
            )
            simulation_errors.append(None)
        except Exception as error:
            simulation_results.append(None)
            simulation_errors.append(f"{type(error).__name__}: {error}")

    return build_pareto_fitness_records(
        genomes,
        simulation_results,
        simulation_errors,
        latency_weight=normalized_latency_weight,
        cost_weight=normalized_cost_weight,
    )


def build_pareto_fitness_records(
    genomes: Sequence[ArchitectureGenome],
    simulation_results: Sequence[TrafficSimulationResult | None],
    simulation_errors: Sequence[str | None] | None = None,
    *,
    latency_weight: float = DEFAULT_LATENCY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
) -> list[FitnessRecord]:
    """Build and rank fitness records from already evaluated simulations.

    This function lets callers such as :class:`EvolutionEngine` evaluate
    simulations concurrently without duplicating NSGA-II ranking logic.
    """
    if len(genomes) != len(simulation_results):
        raise ValueError("genomes and simulation_results must have equal lengths")

    errors = (
        list(simulation_errors)
        if simulation_errors is not None
        else [None] * len(genomes)
    )
    if len(errors) != len(genomes):
        raise ValueError("simulation_errors must align with genomes")
    normalized_latency_weight, normalized_cost_weight = _validate_objective_weights(
        latency_weight,
        cost_weight,
    )

    records: list[FitnessRecord] = []
    for index, (genome, result, error) in enumerate(
        zip(genomes, simulation_results, errors, strict=True)
    ):
        records.append(_build_record(index, genome, result, error))

    for front_rank, front in enumerate(_non_dominated_fronts(records)):
        for index in front:
            records[index]["front_rank"] = front_rank
        _assign_crowding_distance(records, front)

    for record in records:
        record["composite_fitness"] = _composite_fitness(
            record,
            normalized_latency_weight,
            normalized_cost_weight,
        )

    return records


def _build_record(
    genome_index: int,
    genome: ArchitectureGenome,
    result: TrafficSimulationResult | None,
    error: str | None,
) -> FitnessRecord:
    if result is None:
        return {
            "genome_index": genome_index,
            "total_p99_latency_ms": None,
            "total_cost_hourly": _resource_cost(genome),
            "queue_saturation": {},
            "service_metrics": {},
            "chaos_mode": False,
            "chaos_failed_services": (),
            "simulation_error": error or "simulation did not return a result",
            "structural_penalties": SIMULATION_FAILURE_PENALTY,
            "front_rank": -1,
            "crowding_distance": 0.0,
            "composite_fitness": 0.0,
        }

    queue_saturation = dict(result["queue_saturation"])
    structural_penalties = SATURATION_PENALTY * len(queue_saturation)
    if error is not None:
        structural_penalties += SIMULATION_FAILURE_PENALTY

    return {
        "genome_index": genome_index,
        "total_p99_latency_ms": result["total_p99_latency_ms"],
        "total_cost_hourly": result["total_cost_hourly"],
        "queue_saturation": queue_saturation,
        "service_metrics": dict(result["service_metrics"]),
        "chaos_mode": result["chaos_mode"],
        "chaos_failed_services": result["chaos_failed_services"],
        "simulation_error": error,
        "structural_penalties": structural_penalties,
        "front_rank": -1,
        "crowding_distance": 0.0,
        "composite_fitness": 0.0,
    }


def _non_dominated_fronts(records: Sequence[FitnessRecord]) -> list[list[int]]:
    """Partition records into NSGA-II fronts using latency and cost minimization."""
    dominated_by: list[list[int]] = [[] for _ in records]
    domination_count = [0 for _ in records]
    first_front: list[int] = []

    for candidate_index, candidate in enumerate(records):
        for opponent_index, opponent in enumerate(records):
            if candidate_index == opponent_index:
                continue
            if _dominates(candidate, opponent):
                dominated_by[candidate_index].append(opponent_index)
            elif _dominates(opponent, candidate):
                domination_count[candidate_index] += 1
        if domination_count[candidate_index] == 0:
            first_front.append(candidate_index)

    fronts: list[list[int]] = []
    current_front = first_front
    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for candidate_index in current_front:
            for dominated_index in dominated_by[candidate_index]:
                domination_count[dominated_index] -= 1
                if domination_count[dominated_index] == 0:
                    next_front.append(dominated_index)
        current_front = next_front

    return fronts


def _dominates(left: FitnessRecord, right: FitnessRecord) -> bool:
    """Return constrained NSGA-II dominance for two minimization goals.

    A finite-latency architecture always dominates an architecture with queue
    saturation. When both are infeasible, the one with the smaller aggregate
    overload dominates before cost is considered. This prevents a cheaper but
    more overloaded topology from becoming the preferred parent.
    """
    left_violation = _constraint_violation(left)
    right_violation = _constraint_violation(right)
    if left_violation == 0.0 and right_violation > 0.0:
        return True
    if left_violation > 0.0 and right_violation == 0.0:
        return False
    if left_violation != right_violation:
        return left_violation < right_violation

    left_latency, left_cost = _objectives(left)
    right_latency, right_cost = _objectives(right)
    return (
        left_latency <= right_latency
        and left_cost <= right_cost
        and (left_latency < right_latency or left_cost < right_cost)
    )


def _assign_crowding_distance(
    records: list[FitnessRecord],
    front: Sequence[int],
) -> None:
    """Assign NSGA-II crowding distances without non-finite arithmetic leaks."""
    for index in front:
        records[index]["crowding_distance"] = 0.0

    if len(front) <= 2:
        for index in front:
            records[index]["crowding_distance"] = inf
        return

    for objective_index in range(2):
        ordered = sorted(
            front,
            key=lambda index: (_objectives(records[index])[objective_index], index),
        )
        records[ordered[0]]["crowding_distance"] = inf
        records[ordered[-1]]["crowding_distance"] = inf

        lower = _objectives(records[ordered[0]])[objective_index]
        upper = _objectives(records[ordered[-1]])[objective_index]
        if not isfinite(lower) or not isfinite(upper) or upper == lower:
            continue

        for position in range(1, len(ordered) - 1):
            current = records[ordered[position]]
            if not isfinite(current["crowding_distance"]):
                continue

            previous = _objectives(records[ordered[position - 1]])[objective_index]
            following = _objectives(records[ordered[position + 1]])[objective_index]
            if not isfinite(previous) or not isfinite(following):
                continue
            current["crowding_distance"] += (following - previous) / (upper - lower)


def _composite_fitness(
    record: FitnessRecord,
    latency_weight: float,
    cost_weight: float,
) -> float:
    """Calculate the dashboard scalar while preserving Pareto ranking separately."""
    latency, cost = _objectives(record)
    denominator = (
        latency_weight * latency
        + cost_weight * cost
        + record["structural_penalties"]
    )
    if not isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return 1000.0 / denominator


def _objectives(record: FitnessRecord) -> tuple[float, float]:
    """Map nullable output metrics to safe minimization objectives."""
    latency = record["total_p99_latency_ms"]
    return (inf if latency is None else latency, record["total_cost_hourly"])


def _constraint_violation(record: FitnessRecord) -> float:
    """Measure queue overload for feasibility-first NSGA-II dominance."""
    if record["simulation_error"] is not None:
        return inf
    violation = 0.0
    for details in record["queue_saturation"].values():
        capacity = details["capacity_qps"]
        if capacity <= 0.0:
            return inf
        violation += 1.0 + max(0.0, details["excess_qps"]) / capacity
    return violation


def _resource_cost(genome: ArchitectureGenome) -> float:
    """Calculate cost for simulations that failed before reporting metrics."""
    return sum(
        service.replicas
        * (service.cpu_limit * 0.04 + (service.mem_limit_mb / 1024.0) * 0.01)
        for service in genome.services.values()
    )


def _validate_inputs(
    genomes: Sequence[ArchitectureGenome],
    simulator: TrafficSimulator,
    baseline_qps: float,
) -> None:
    """Reject malformed evaluation inputs before processing a population."""
    if not isinstance(simulator, TrafficSimulator):
        raise TypeError("simulator must be a TrafficSimulator")
    if any(not isinstance(genome, ArchitectureGenome) for genome in genomes):
        raise TypeError("genomes must contain only ArchitectureGenome instances")
    if isinstance(baseline_qps, bool) or not isinstance(baseline_qps, Real):
        raise ValueError("baseline_qps must be a real number")
    if not isfinite(float(baseline_qps)) or baseline_qps <= 0.0:
        raise ValueError("baseline_qps must be finite and greater than zero")


def _validate_objective_weights(
    latency_weight: float,
    cost_weight: float,
) -> tuple[float, float]:
    """Validate a finite convex weighting for dashboard fitness ordering."""
    weights = (latency_weight, cost_weight)
    if any(isinstance(weight, bool) or not isinstance(weight, Real) for weight in weights):
        raise ValueError("objective weights must be real numbers")

    normalized_weights = tuple(float(weight) for weight in weights)
    if any(
        not isfinite(weight) or not 0.0 <= weight <= 1.0
        for weight in normalized_weights
    ):
        raise ValueError("objective weights must be finite values between zero and one")
    if abs(sum(normalized_weights) - 1.0) > 1e-6:
        raise ValueError("latency_weight and cost_weight must sum to 1.0")
    return normalized_weights
