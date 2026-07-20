"""Deterministic Docker Compose and Kubernetes topology extraction."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

from evoarch.models.genome import ArchitectureGenome, EdgeGene, ServiceGene

DEFAULT_CPU_LIMIT = 0.5
DEFAULT_MEMORY_LIMIT_MB = 512
DEFAULT_REPLICAS = 1
DEFAULT_ROUTING_ALGORITHM = "round_robin"
DEFAULT_EDGE_LATENCY_MS = 2.0
_WORKLOAD_KINDS = frozenset({"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"})
_ROUTING_ALGORITHMS = frozenset({"round_robin", "least_connections", "random"})
_MEMORY_PATTERN = re.compile(
    r"^(?P<quantity>\d+(?:\.\d+)?)(?P<unit>k|kb|ki|kib|m|mb|mi|mib|g|gb|gi|gib|t|tb|ti|tib)?$",
    re.IGNORECASE,
)


class InfrastructureParseError(ValueError):
    """Raised when an infrastructure document cannot form an EvoArch topology."""


def parse_infrastructure_to_genome(file_content: str) -> ArchitectureGenome:
    """Parse supported YAML into a bounded, directed :class:`ArchitectureGenome`."""
    if not isinstance(file_content, str):
        raise TypeError("file_content must be a string")
    if not file_content.strip():
        raise InfrastructureParseError("infrastructure file cannot be empty")

    try:
        documents = [
            document
            for document in yaml.safe_load_all(file_content)
            if isinstance(document, Mapping)
        ]
    except yaml.YAMLError as error:
        raise InfrastructureParseError("infrastructure file is not valid YAML") from error

    if not documents:
        raise InfrastructureParseError("infrastructure file must contain YAML mappings")

    compose_root = documents[0] if len(documents) == 1 else None
    if isinstance(compose_root, Mapping) and isinstance(compose_root.get("services"), Mapping):
        return _parse_compose(compose_root)
    return _parse_kubernetes(documents)


def _parse_compose(compose_root: Mapping[str, Any]) -> ArchitectureGenome:
    services_value = compose_root.get("services")
    if not isinstance(services_value, Mapping) or not services_value:
        raise InfrastructureParseError("Docker Compose file must declare at least one service")

    services: dict[str, ServiceGene] = {}
    configs: dict[str, Mapping[str, Any]] = {}
    for raw_name, raw_config in services_value.items():
        service_name = _service_name(raw_name)
        if not isinstance(raw_config, Mapping):
            raise InfrastructureParseError(f"Compose service {service_name!r} must be a mapping")
        configs[service_name] = raw_config
        services[service_name] = _compose_service_gene(service_name, raw_config)

    candidates: list[tuple[str, str]] = []
    service_names = tuple(services)
    for service_name, config in configs.items():
        for dependency_name in _compose_dependencies(config):
            candidates.append((service_name, dependency_name))
        for value in _environment_values(config.get("environment")):
            candidates.extend(
                (service_name, dependency_name)
                for dependency_name in _referenced_services(value, service_names)
            )

    return ArchitectureGenome(services=services, edges=_acyclic_edges(candidates, services))


def _compose_service_gene(
    service_name: str,
    config: Mapping[str, Any],
) -> ServiceGene:
    deploy = _mapping(config.get("deploy"))
    resources = _mapping(deploy.get("resources"))
    limits = _mapping(resources.get("limits"))
    labels = _labels(config.get("labels"))
    return ServiceGene(
        service_name=service_name,
        replicas=_bounded_replicas(deploy.get("replicas")),
        cpu_limit=_bounded_cpu(_cpu_value(limits.get("cpus")), DEFAULT_CPU_LIMIT),
        mem_limit_mb=_bounded_memory(
            _memory_value(limits.get("memory")),
            DEFAULT_MEMORY_LIMIT_MB,
        ),
        routing_algorithm=_routing_algorithm(labels),
    )


def _parse_kubernetes(documents: Sequence[Mapping[str, Any]]) -> ArchitectureGenome:
    services: dict[str, ServiceGene] = {}
    environment_by_service: dict[str, list[str]] = {}
    service_resource_names: list[str] = []

    for document in documents:
        kind = _string(document.get("kind"))
        metadata = _mapping(document.get("metadata"))
        service_name = _service_name(metadata.get("name")) if metadata.get("name") else ""
        if not service_name:
            continue
        if kind in _WORKLOAD_KINDS:
            pod_template = _pod_template(document, kind)
            pod_spec = _mapping(pod_template.get("spec"))
            containers = _sequence_of_mappings(pod_spec.get("containers"))
            metadata_labels = _labels(metadata.get("labels"))
            template_metadata = _mapping(pod_template.get("metadata"))
            template_labels = _labels(template_metadata.get("labels"))
            services[service_name] = ServiceGene(
                service_name=service_name,
                replicas=_kubernetes_replicas(document, kind),
                cpu_limit=_container_cpu_limit(containers),
                mem_limit_mb=_container_memory_limit(containers),
                routing_algorithm=_routing_algorithm({**metadata_labels, **template_labels}),
            )
            environment_by_service[service_name] = _container_environment_values(containers)
        elif kind == "Service":
            service_resource_names.append(service_name)

    for service_name in service_resource_names:
        services.setdefault(
            service_name,
            ServiceGene(
                service_name=service_name,
                replicas=DEFAULT_REPLICAS,
                cpu_limit=DEFAULT_CPU_LIMIT,
                mem_limit_mb=DEFAULT_MEMORY_LIMIT_MB,
                routing_algorithm=DEFAULT_ROUTING_ALGORITHM,
            ),
        )

    if not services:
        raise InfrastructureParseError(
            "Kubernetes YAML must declare a supported workload or Service resource"
        )

    candidates: list[tuple[str, str]] = []
    service_names = tuple(services)
    for source, environment_values in environment_by_service.items():
        for value in environment_values:
            candidates.extend(
                (source, target)
                for target in _referenced_services(value, service_names)
            )
    return ArchitectureGenome(services=services, edges=_acyclic_edges(candidates, services))


def _pod_template(document: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    spec = _mapping(document.get("spec"))
    if kind == "CronJob":
        job_template = _mapping(spec.get("jobTemplate"))
        job_spec = _mapping(job_template.get("spec"))
        return _mapping(job_spec.get("template"))
    return _mapping(spec.get("template"))


def _kubernetes_replicas(document: Mapping[str, Any], kind: str) -> int:
    if kind in {"DaemonSet", "Job", "CronJob"}:
        return DEFAULT_REPLICAS
    return _bounded_replicas(_mapping(document.get("spec")).get("replicas"))


def _container_cpu_limit(containers: Sequence[Mapping[str, Any]]) -> float:
    values = [
        parsed
        for container in containers
        if (parsed := _cpu_value(_mapping(_mapping(container.get("resources")).get("limits")).get("cpu")))
        is not None
    ]
    return _bounded_cpu(sum(values) if values else None, DEFAULT_CPU_LIMIT)


def _container_memory_limit(containers: Sequence[Mapping[str, Any]]) -> int:
    values = [
        parsed
        for container in containers
        if (parsed := _memory_value(_mapping(_mapping(container.get("resources")).get("limits")).get("memory")))
        is not None
    ]
    return _bounded_memory(sum(values) if values else None, DEFAULT_MEMORY_LIMIT_MB)


def _compose_dependencies(config: Mapping[str, Any]) -> list[str]:
    dependencies: list[str] = []
    depends_on = config.get("depends_on")
    if isinstance(depends_on, Mapping):
        dependencies.extend(_service_name(value) for value in depends_on)
    elif isinstance(depends_on, Sequence) and not isinstance(depends_on, str):
        dependencies.extend(_service_name(value) for value in depends_on)
    elif isinstance(depends_on, str):
        dependencies.append(_service_name(depends_on))

    links = config.get("links")
    if isinstance(links, Sequence) and not isinstance(links, str):
        for link in links:
            if isinstance(link, str):
                dependencies.append(_service_name(link.split(":", maxsplit=1)[0]))
    return [dependency for dependency in dependencies if dependency]


def _environment_values(value: object) -> list[str]:
    if isinstance(value, Mapping):
        return [_string(item) for item in value.values() if isinstance(item, (str, int, float))]
    if isinstance(value, Sequence) and not isinstance(value, str):
        values: list[str] = []
        for item in value:
            if isinstance(item, str):
                values.append(item.split("=", maxsplit=1)[-1])
        return values
    return []


def _container_environment_values(containers: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for container in containers:
        environment = container.get("env")
        if not isinstance(environment, Sequence) or isinstance(environment, str):
            continue
        for entry in environment:
            if isinstance(entry, Mapping) and isinstance(entry.get("value"), str):
                values.append(entry["value"])
    return values


def _referenced_services(value: str, service_names: Sequence[str]) -> list[str]:
    references: list[str] = []
    for service_name in sorted(service_names, key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_.-]){re.escape(service_name)}(?=$|[:/,;\s])"
        if re.search(pattern, value):
            references.append(service_name)
    return references


def _acyclic_edges(
    candidates: Sequence[tuple[str, str]],
    services: Mapping[str, ServiceGene],
) -> list[EdgeGene]:
    adjacency: dict[str, set[str]] = {service_name: set() for service_name in services}
    edges: list[EdgeGene] = []
    seen: set[tuple[str, str]] = set()
    for source, target in candidates:
        if source not in services or target not in services or source == target:
            continue
        edge_key = (source, target)
        if edge_key in seen or _has_path(adjacency, target, source):
            continue
        seen.add(edge_key)
        adjacency[source].add(target)
        edges.append(
            EdgeGene(
                source=source,
                target=target,
                base_latency_ms=DEFAULT_EDGE_LATENCY_MS,
            )
        )
    return edges


def _has_path(adjacency: Mapping[str, set[str]], source: str, target: str) -> bool:
    pending = [source]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency.get(current, set()) - visited)
    return False


def _routing_algorithm(labels: Mapping[str, str]) -> str:
    for key in ("evoarch.io/routing-algorithm", "routing_algorithm", "routing"):
        value = labels.get(key)
        if value:
            normalized = value.strip().lower().replace("-", "_")
            if normalized in _ROUTING_ALGORITHMS:
                return normalized
    return DEFAULT_ROUTING_ALGORITHM


def _labels(value: object) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {
            _string(key): _string(item)
            for key, item in value.items()
            if _string(key) and _string(item)
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        labels: dict[str, str] = {}
        for item in value:
            if isinstance(item, str) and "=" in item:
                key, label_value = item.split("=", maxsplit=1)
                labels[key.strip()] = label_value.strip()
        return labels
    return {}


def _bounded_replicas(value: object) -> int:
    try:
        replicas = int(value) if value is not None else DEFAULT_REPLICAS
    except (TypeError, ValueError):
        replicas = DEFAULT_REPLICAS
    return max(1, min(20, replicas))


def _bounded_cpu(value: float | None, default: float) -> float:
    cpu_limit = default if value is None else value
    return round(max(0.1, min(4.0, cpu_limit)), 3)


def _bounded_memory(value: int | None, default: int) -> int:
    memory_limit = default if value is None else value
    return max(128, min(8192, int(round(memory_limit))))


def _cpu_value(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    raw_value = value.strip().lower()
    try:
        if raw_value.endswith("m"):
            return float(raw_value[:-1]) / 1000.0
        return float(raw_value)
    except ValueError:
        return None


def _memory_value(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    if not isinstance(value, str):
        return None
    match = _MEMORY_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    quantity = float(match.group("quantity"))
    unit = (match.group("unit") or "m").lower()
    multipliers = {
        "k": 0.001,
        "kb": 0.001,
        "ki": 1.0 / 1024.0,
        "kib": 1.0 / 1024.0,
        "m": 1.0,
        "mb": 1.0,
        "mi": 1.048576,
        "mib": 1.048576,
        "g": 1000.0,
        "gb": 1000.0,
        "gi": 1024.0,
        "gib": 1024.0,
        "t": 1_000_000.0,
        "tb": 1_000_000.0,
        "ti": 1_048_576.0,
        "tib": 1_048_576.0,
    }
    return int(round(quantity * multipliers[unit]))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _service_name(value: object) -> str:
    return _string(value).strip()


def _string(value: object) -> str:
    return value if isinstance(value, str) else str(value) if value is not None else ""
