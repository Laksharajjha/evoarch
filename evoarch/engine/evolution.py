"""Parallel NSGA-II evolutionary execution for EvoArch genomes."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from math import ceil, inf, isfinite
from numbers import Real
from random import Random
from typing import Literal, Sequence

from evoarch.models.genome import ArchitectureGenome, ServiceGene
from evoarch.optimizer.fitness import FitnessRecord, build_pareto_fitness_records
from evoarch.simulation.traffic import (
    ChaosScenario,
    TrafficSimulationResult,
    TrafficSimulator,
)

MutationStrategy = Literal[
    "scale_replicas",
    "adjust_resources",
    "toggle_routing",
    "break_bottleneck",
]


class EvolutionEngine:
    """Create successive architecture populations using NSGA-II selection.

    Population simulations run in a thread pool. Each simulation is independent
    and ``TrafficSimulator`` has no mutable evaluation state, so parallel runs
    preserve deterministic metrics while the engine's seeded RNG controls all
    crossover and mutation choices.
    """

    _ROUTING_ALGORITHMS = ("round_robin", "least_connections", "random")
    _MUTATION_STRATEGIES: tuple[MutationStrategy, ...] = (
        "scale_replicas",
        "adjust_resources",
        "toggle_routing",
        "break_bottleneck",
    )

    def __init__(
        self,
        simulator: TrafficSimulator,
        baseline_qps: float,
        *,
        elite_count: int = 2,
        tournament_size: int = 3,
        mutation_rate: float = 0.35,
        max_workers: int | None = None,
        random_seed: int | None = None,
        latency_weight: float = 0.7,
        cost_weight: float = 0.3,
        max_replicas_cap: int = 20,
        chaos_mode: bool = False,
    ) -> None:
        """Configure a reproducible evolutionary execution loop.

        When Chaos Mode is enabled, every generation evaluates all candidates
        against one shared 10--20% pod-failure scenario. A selected service loses
        one replica, so singleton services fail while redundant services retain
        reduced but measurable capacity.
        """
        if not isinstance(simulator, TrafficSimulator):
            raise TypeError("simulator must be a TrafficSimulator")
        self._simulator = simulator
        self._baseline_qps = self._validate_positive_real(
            baseline_qps,
            "baseline_qps",
        )
        self._elite_count = self._validate_non_negative_int(elite_count, "elite_count")
        self._tournament_size = self._validate_positive_int(
            tournament_size,
            "tournament_size",
        )
        self._mutation_rate = self._validate_probability(mutation_rate)
        self._max_workers = self._validate_max_workers(max_workers)
        seed = self._validate_random_seed(random_seed)
        self._random = Random(seed)
        self._chaos_random = Random(None if seed is None else seed ^ 0x5A17)
        self._latency_weight, self._cost_weight = self._validate_objective_weights(
            latency_weight,
            cost_weight,
        )
        self._max_replicas_cap = self._validate_max_replicas_cap(max_replicas_cap)
        self._chaos_mode = self._validate_bool(chaos_mode, "chaos_mode")
        self.last_fitness: list[FitnessRecord] = []
        self.last_chaos_scenario: ChaosScenario | None = None

    def run_generation(
        self,
        population: list[ArchitectureGenome],
    ) -> list[ArchitectureGenome]:
        """Evaluate a population in parallel and return its next generation."""
        if not population:
            self.last_fitness = []
            return []
        if any(not isinstance(genome, ArchitectureGenome) for genome in population):
            raise TypeError("population must contain only ArchitectureGenome instances")

        fitness_records = self._evaluate_population_parallel(population)
        ranked_indices = sorted(
            range(len(population)),
            key=lambda index: self._selection_key(fitness_records[index]),
        )
        elite_count = min(self._elite_count, len(population))
        next_population = [
            self._cap_genome_replicas(population[index])
            for index in ranked_indices[:elite_count]
        ]

        while len(next_population) < len(population):
            first_parent_index = self._tournament_select(fitness_records)
            second_parent_index = self._tournament_select(fitness_records)
            child = self.crossover(
                population[first_parent_index],
                population[second_parent_index],
            )
            if self._random.random() < self._mutation_rate:
                child = self.mutate(child, fitness_records[first_parent_index])
            next_population.append(child)

        return next_population

    def crossover(
        self,
        first_parent: ArchitectureGenome,
        second_parent: ArchitectureGenome,
    ) -> ArchitectureGenome:
        """Blend shared service allocations while retaining a valid base topology."""
        if not isinstance(first_parent, ArchitectureGenome) or not isinstance(
            second_parent,
            ArchitectureGenome,
        ):
            raise TypeError("parents must be ArchitectureGenome instances")

        base_parent, donor_parent = (
            (first_parent, second_parent)
            if self._random.random() < 0.5
            else (second_parent, first_parent)
        )
        child_services: dict[str, ServiceGene] = {}
        for service_name, base_service in base_parent.services.items():
            donor_service = donor_parent.services.get(service_name)
            if donor_service is None:
                child_services[service_name] = self._cap_service_replicas(base_service)
                continue
            child_services[service_name] = self._blend_service_genes(
                base_service,
                donor_service,
            )

        return ArchitectureGenome(
            services=child_services,
            edges=[edge.model_copy(deep=True) for edge in base_parent.edges],
        )

    def mutate(
        self,
        genome: ArchitectureGenome,
        fitness_record: FitnessRecord | None = None,
        *,
        strategy: MutationStrategy | None = None,
    ) -> ArchitectureGenome:
        """Apply one validated, discrete mutation without changing ``genome``."""
        if not isinstance(genome, ArchitectureGenome):
            raise TypeError("genome must be an ArchitectureGenome")
        selected_strategy = strategy or self._random.choice(self._MUTATION_STRATEGIES)
        if strategy is None and fitness_record is not None and fitness_record[
            "queue_saturation"
        ]:
            selected_strategy = "break_bottleneck"
        if selected_strategy not in self._MUTATION_STRATEGIES:
            raise ValueError(f"unsupported mutation strategy: {selected_strategy!r}")

        services = {
            service_name: self._cap_service_replicas(service)
            for service_name, service in genome.services.items()
        }
        if selected_strategy == "scale_replicas":
            self._scale_replicas(services)
        elif selected_strategy == "adjust_resources":
            self._adjust_resources(services)
        elif selected_strategy == "toggle_routing":
            self._toggle_routing(services)
        else:
            self._break_bottleneck(services, fitness_record)

        return ArchitectureGenome(
            services=services,
            edges=[edge.model_copy(deep=True) for edge in genome.edges],
        )

    def _evaluate_population_parallel(
        self,
        population: Sequence[ArchitectureGenome],
    ) -> list[FitnessRecord]:
        """Run all workload simulations concurrently and preserve input ordering."""
        simulation_results: list[TrafficSimulationResult | None] = [None] * len(
            population
        )
        simulation_errors: list[str | None] = [None] * len(population)
        worker_count = self._max_workers or min(32, len(population))
        chaos_scenario = (
            ChaosScenario.sample(population[0].services.keys(), self._chaos_random)
            if self._chaos_mode
            else None
        )
        self.last_chaos_scenario = chaos_scenario

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="evoarch-eval",
        ) as executor:
            futures: dict[Future[TrafficSimulationResult], int] = {
                executor.submit(
                    self._simulator.simulate_load,
                    genome,
                    self._baseline_qps,
                    chaos_mode=self._chaos_mode,
                    chaos_scenario=chaos_scenario,
                ): index
                for index, genome in enumerate(population)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    simulation_results[index] = future.result()
                except Exception as error:
                    simulation_errors[index] = f"{type(error).__name__}: {error}"

        self.last_fitness = build_pareto_fitness_records(
            population,
            simulation_results,
            simulation_errors,
            latency_weight=self._latency_weight,
            cost_weight=self._cost_weight,
        )
        return self.last_fitness

    def _tournament_select(self, fitness_records: Sequence[FitnessRecord]) -> int:
        """Select the best sampled candidate by rank, diversity, then fitness."""
        sample_size = min(self._tournament_size, len(fitness_records))
        candidates = self._random.sample(range(len(fitness_records)), sample_size)
        return min(candidates, key=lambda index: self._selection_key(fitness_records[index]))

    @staticmethod
    def _selection_key(record: FitnessRecord) -> tuple[int, float, float, float, int]:
        """Produce a stable minimization key for NSGA-II tournament selection."""
        latency = record["total_p99_latency_ms"]
        crowding_distance = record["crowding_distance"]
        normalized_crowding = (
            crowding_distance if isfinite(crowding_distance) else inf
        )
        return (
            record["front_rank"],
            -normalized_crowding,
            -record["composite_fitness"],
            inf if latency is None else latency,
            record["genome_index"],
        )

    def _blend_service_genes(
        self,
        first_service: ServiceGene,
        second_service: ServiceGene,
    ) -> ServiceGene:
        """Create a bounded allocation blend for services common to both parents."""
        replicas = self._clamp_int(
            round((first_service.replicas + second_service.replicas) / 2),
            1,
            self._max_replicas_cap,
        )
        cpu_limit = round(
            self._clamp_float(
                (first_service.cpu_limit + second_service.cpu_limit) / 2.0,
                0.1,
                4.0,
            ),
            3,
        )
        mem_limit_mb = self._clamp_int(
            round((first_service.mem_limit_mb + second_service.mem_limit_mb) / 2),
            128,
            8192,
        )
        routing_algorithm = self._random.choice(
            (first_service.routing_algorithm, second_service.routing_algorithm)
        )
        return ServiceGene(
            service_name=first_service.service_name,
            replicas=replicas,
            cpu_limit=cpu_limit,
            mem_limit_mb=mem_limit_mb,
            routing_algorithm=routing_algorithm,
        )

    def _scale_replicas(self, services: dict[str, ServiceGene]) -> None:
        """Increase or decrease one service replica count within schema bounds."""
        service_name = self._random.choice(tuple(services))
        service = services[service_name]
        delta = self._random.choice((-1, 1))
        replicas = self._clamp_int(service.replicas + delta, 1, self._max_replicas_cap)
        if replicas == service.replicas:
            replicas = self._clamp_int(
                service.replicas - delta,
                1,
                self._max_replicas_cap,
            )
        services[service_name] = self._replace_service(service, replicas=replicas)

    def _adjust_resources(self, services: dict[str, ServiceGene]) -> None:
        """Apply bounded CPU and memory resource steps to one service."""
        service_name = self._random.choice(tuple(services))
        service = services[service_name]
        cpu_delta = self._random.choice((-0.25, 0.25))
        cpu_limit = round(
            self._clamp_float(service.cpu_limit + cpu_delta, 0.1, 4.0),
            3,
        )
        if cpu_limit == service.cpu_limit:
            cpu_limit = round(
                self._clamp_float(service.cpu_limit - cpu_delta, 0.1, 4.0),
                3,
            )
        memory_delta = self._random.choice((-256, 256))
        mem_limit_mb = self._clamp_int(
            service.mem_limit_mb + memory_delta,
            128,
            8192,
        )
        if mem_limit_mb == service.mem_limit_mb:
            mem_limit_mb = self._clamp_int(
                service.mem_limit_mb - memory_delta,
                128,
                8192,
            )
        services[service_name] = self._replace_service(
            service,
            cpu_limit=cpu_limit,
            mem_limit_mb=mem_limit_mb,
        )

    def _toggle_routing(self, services: dict[str, ServiceGene]) -> None:
        """Swap one service to a different supported load-balancing algorithm."""
        service_name = self._random.choice(tuple(services))
        service = services[service_name]
        alternatives = tuple(
            algorithm
            for algorithm in self._ROUTING_ALGORITHMS
            if algorithm != service.routing_algorithm
        )
        services[service_name] = self._replace_service(
            service,
            routing_algorithm=self._random.choice(alternatives),
        )

    def _break_bottleneck(
        self,
        services: dict[str, ServiceGene],
        fitness_record: FitnessRecord | None,
    ) -> None:
        """Scale the most saturated or highly utilized service first."""
        service_name = self._bottleneck_service_name(services, fitness_record)
        service = services[service_name]
        metrics = (
            fitness_record["service_metrics"].get(service_name)
            if fitness_record is not None
            else None
        )
        if metrics is not None:
            service_rate = metrics["service_rate_per_replica_qps"]
            arrival_rate = metrics["arrival_rate_qps"]
            target_utilization = 0.85
            failure_reserve = 1 if metrics["chaos_failure_injected"] else 0
            required_replicas = failure_reserve + ceil(
                arrival_rate / (service_rate * target_utilization)
            )
            target_replicas = min(self._max_replicas_cap, required_replicas)
            if target_replicas > service.replicas:
                services[service_name] = self._replace_service(
                    service,
                    replicas=target_replicas,
                )
                return

            base_rate_per_cpu = service_rate / service.cpu_limit
            active_replicas = max(1, service.replicas - failure_reserve)
            required_cpu = arrival_rate / (
                active_replicas * base_rate_per_cpu * target_utilization
            )
            target_cpu = round(
                self._clamp_float(max(service.cpu_limit + 0.25, required_cpu), 0.1, 4.0),
                3,
            )
            if target_cpu > service.cpu_limit:
                services[service_name] = self._replace_service(
                    service,
                    cpu_limit=target_cpu,
                )
                return

        if service.replicas < self._max_replicas_cap:
            services[service_name] = self._replace_service(
                service,
                replicas=service.replicas + 1,
            )
            return
        if service.cpu_limit < 4.0:
            services[service_name] = self._replace_service(
                service,
                cpu_limit=round(min(4.0, service.cpu_limit + 0.25), 3),
            )

    def _bottleneck_service_name(
        self,
        services: dict[str, ServiceGene],
        fitness_record: FitnessRecord | None,
    ) -> str:
        """Find a mutable service targeted by saturation or utilization evidence."""
        if fitness_record is not None:
            saturated_services = [
                service_name
                for service_name in fitness_record["queue_saturation"]
                if service_name in services
            ]
            if saturated_services:
                return max(
                    saturated_services,
                    key=lambda service_name: self._safe_utilization(
                        fitness_record["queue_saturation"][service_name]["utilization"]
                    ),
                )

            measured_services = [
                service_name
                for service_name in fitness_record["service_metrics"]
                if service_name in services
            ]
            if measured_services:
                return max(
                    measured_services,
                    key=lambda service_name: self._safe_utilization(
                        fitness_record["service_metrics"][service_name]["utilization"]
                    ),
                )

        return self._random.choice(tuple(services))

    @staticmethod
    def _replace_service(
        service: ServiceGene,
        **updates: object,
    ) -> ServiceGene:
        """Create a validated replacement instead of mutating a parent gene."""
        values = service.model_dump()
        values.update(updates)
        return ServiceGene(**values)

    def _cap_genome_replicas(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Copy a genome while enforcing the intent-derived replica ceiling."""
        return ArchitectureGenome(
            services={
                service_name: self._cap_service_replicas(service)
                for service_name, service in genome.services.items()
            },
            edges=[edge.model_copy(deep=True) for edge in genome.edges],
        )

    def _cap_service_replicas(self, service: ServiceGene) -> ServiceGene:
        """Copy one service gene with replicas constrained to the active ceiling."""
        return self._replace_service(
            service,
            replicas=min(service.replicas, self._max_replicas_cap),
        )

    @staticmethod
    def _clamp_int(value: int, lower: int, upper: int) -> int:
        """Clamp an integer without changing its type."""
        return max(lower, min(upper, value))

    @staticmethod
    def _clamp_float(value: float, lower: float, upper: float) -> float:
        """Clamp a float without changing its type."""
        return max(lower, min(upper, value))

    @staticmethod
    def _safe_utilization(value: float | None) -> float:
        """Prioritize zero-capacity chaos failures when selecting bottlenecks."""
        return inf if value is None else value

    @staticmethod
    def _validate_positive_real(value: float, field_name: str) -> float:
        """Validate finite, positive numeric settings."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a real number")
        numeric_value = float(value)
        if not isfinite(numeric_value) or numeric_value <= 0.0:
            raise ValueError(f"{field_name} must be finite and greater than zero")
        return numeric_value

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> int:
        """Validate integer settings that must be greater than zero."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be an integer greater than zero")
        return value

    @staticmethod
    def _validate_non_negative_int(value: int, field_name: str) -> int:
        """Validate integer settings that may be zero."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
        return value

    @staticmethod
    def _validate_probability(value: float) -> float:
        """Validate a finite mutation probability in the closed unit interval."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("mutation_rate must be a real number")
        probability = float(value)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("mutation_rate must be between zero and one")
        return probability

    @staticmethod
    def _validate_objective_weights(
        latency_weight: float,
        cost_weight: float,
    ) -> tuple[float, float]:
        """Validate the convex weights used for scalar Pareto tie-breaking."""
        weights = (latency_weight, cost_weight)
        if any(
            isinstance(weight, bool) or not isinstance(weight, Real)
            for weight in weights
        ):
            raise ValueError("objective weights must be real numbers")

        normalized_weights = tuple(float(weight) for weight in weights)
        if any(
            not isfinite(weight) or not 0.0 <= weight <= 1.0
            for weight in normalized_weights
        ):
            raise ValueError(
                "objective weights must be finite values between zero and one"
            )
        if abs(sum(normalized_weights) - 1.0) > 1e-6:
            raise ValueError("latency_weight and cost_weight must sum to 1.0")
        return normalized_weights

    @staticmethod
    def _validate_max_replicas_cap(value: int) -> int:
        """Validate the intent-derived service replica ceiling."""
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
            raise ValueError("max_replicas_cap must be an integer between 1 and 20")
        return value

    @staticmethod
    def _validate_max_workers(value: int | None) -> int | None:
        """Validate an optional thread-pool size."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("max_workers must be a positive integer or None")
        return value

    @staticmethod
    def _validate_random_seed(value: int | None) -> int | None:
        """Validate an optional integer seed for reproducible evolution."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("random_seed must be an integer or None")
        return value

    @staticmethod
    def _validate_bool(value: bool, field_name: str) -> bool:
        """Reject truthy non-booleans at the execution boundary."""
        if not isinstance(value, bool):
            raise ValueError(f"{field_name} must be a boolean")
        return value
