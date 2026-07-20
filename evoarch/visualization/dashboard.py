"""Live FastAPI dashboard for EvoArch optimization runs."""

import asyncio
import logging
import os
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import inf, isfinite
from pathlib import Path
from threading import RLock, Thread
from time import sleep
from typing import Any, Literal, Sequence
from uuid import UUID, uuid4

import yaml
from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from evoarch.api.ai_agent import AIAgentError, EvoArchAIAgent, IntentWeights
from evoarch.api.infrastructure_parser import parse_infrastructure_to_genome
from evoarch.engine.evolution import EvolutionEngine, MutationStrategy
from evoarch.models.genome import ArchitectureGenome, EdgeGene, ServiceGene
from evoarch.optimizer.fitness import FitnessRecord, build_pareto_fitness_records
from evoarch.simulation.traffic import ServiceLoadDetails, TrafficSimulator

LOGGER = logging.getLogger(__name__)
load_dotenv()
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"
MAX_TOPOLOGY_UPLOAD_BYTES = 1_000_000
INGESTION_PROBE_QPS = 1.0
EDGE_LABEL_QUEUE_DELAY_THRESHOLD_MS = 1.0
SUPPORTED_TOPOLOGY_SUFFIXES = frozenset({".yaml", ".yml"})
ACCESS_CODE = os.getenv("ACCESS_CODE", "EVO-JUDGE-26")
SESSION_COOKIE_NAME = "evoarch_session"
SESSION_HISTORY_LIMIT = 250
limiter = Limiter(key_func=get_remote_address)


class OptimizationRunRequest(BaseModel):
    """Validated controls for a background optimization run."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_prompt: str = Field(min_length=1, max_length=4_000)
    generation_count: int = Field(default=18, ge=1, le=200)
    population_size: int = Field(default=24, ge=4, le=128)
    baseline_qps: float = Field(default=120.0, gt=0.0, le=100_000.0)
    generation_delay_ms: int = Field(default=350, ge=0, le=5_000)
    random_seed: int | None = None
    chaos_mode: bool = False


class OptimizationRunResponse(BaseModel):
    """Acknowledgement returned immediately after scheduling a run."""

    run_id: str
    status: Literal["started"]


class TopologyUploadResponse(BaseModel):
    """Confirmation returned after an uploaded topology becomes the baseline."""

    status: Literal["ingested"]
    source_file: str
    service_count: int
    edge_count: int


def _require_access_code(x_access_code: str | None) -> None:
    """Reject protected control-plane calls without the configured access code."""
    if x_access_code != ACCESS_CODE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Access DNA",
        )


def _normalize_session_id(candidate: str | None) -> str:
    """Return a canonical UUID session identifier or mint a new one."""
    if candidate is not None:
        try:
            return str(UUID(candidate))
        except (TypeError, ValueError, AttributeError):
            pass
    return str(uuid4())


def _http_session_id(request: Request) -> str:
    """Resolve a session from an opt-in header or the browser session cookie."""
    return _normalize_session_id(
        request.headers.get("X-EvoArch-Session")
        or request.cookies.get(SESSION_COOKIE_NAME)
    )


def _websocket_session_id(websocket: WebSocket) -> str:
    """Resolve a WebSocket workspace without requiring client-side auth changes."""
    return _normalize_session_id(
        websocket.query_params.get("session_id")
        or websocket.cookies.get(SESSION_COOKIE_NAME)
    )


@dataclass
class DashboardSession:
    """Mutable dashboard workspace owned by one browser session."""

    session_id: str
    instance_id: str
    baseline_genome: ArchitectureGenome
    active_run_id: str | None = None
    active_thread: Thread | None = None
    event_loop: asyncio.AbstractEventLoop | None = None
    connections: set[WebSocket] = field(default_factory=set)
    history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=SESSION_HISTORY_LIMIT)
    )


class DashboardState:
    """Coordinate isolated dashboard workspaces and their event streams."""

    def __init__(self, baseline_genome: ArchitectureGenome) -> None:
        self._lock = RLock()
        self._baseline_template = baseline_genome.model_copy(deep=True)
        self._sessions: dict[str, DashboardSession] = {}

    def start_optimization(
        self,
        session_id: str,
        request: OptimizationRunRequest,
        intent_weights: IntentWeights,
    ) -> str:
        """Start one optimization worker inside a session-owned workspace."""
        with self._lock:
            session = self._get_or_create_session_locked(session_id)
            if session.active_run_id is not None:
                raise RuntimeError("an optimization run is already active")
            run_id = uuid4().hex
            session.active_run_id = run_id
            baseline_genome = session.baseline_genome.model_copy(deep=True)
            worker = Thread(
                target=_run_optimization,
                args=(
                    self,
                    session.session_id,
                    session.instance_id,
                    run_id,
                    request.model_copy(deep=True),
                    intent_weights.model_copy(deep=True),
                    baseline_genome,
                ),
                name=f"evoarch-optimization-{run_id[:8]}",
                daemon=True,
            )
            session.active_thread = worker
            session_instance_id = session.instance_id

        self.publish(
            session_id,
            _intent_translated_event(
                run_id,
                intent_weights,
                chaos_mode=request.chaos_mode,
            ),
            session_instance_id=session_instance_id,
        )
        worker.start()
        return run_id

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """Accept a client and replay the recent run history."""
        await websocket.accept()
        with self._lock:
            session = self._get_or_create_session_locked(session_id)
            session.event_loop = asyncio.get_running_loop()
            session.connections.add(websocket)
            history = list(session.history)

        for event in history:
            await websocket.send_json(event)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        """Drop a closed connection and release an empty session workspace."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.connections.discard(websocket)
            if not session.connections:
                self._sessions.pop(session_id, None)

    def publish(
        self,
        session_id: str,
        event: dict[str, Any],
        *,
        session_instance_id: str | None = None,
    ) -> None:
        """Store and broadcast an event only inside its owning session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            if session_instance_id is not None and session.instance_id != session_instance_id:
                return
            session.history.append(event)
            event_loop = session.event_loop
            current_instance_id = session.instance_id

        if event_loop is None or not event_loop.is_running():
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._broadcast(session_id, current_instance_id, event),
                event_loop,
            )
            future.add_done_callback(self._consume_broadcast_result)
        except RuntimeError:
            LOGGER.debug("Dashboard event loop closed before a broadcast could run")

    def finish_optimization(
        self,
        session_id: str,
        session_instance_id: str,
        run_id: str,
    ) -> None:
        """Mark a session worker complete without reviving stale state."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.instance_id != session_instance_id:
                return
            if session.active_run_id == run_id:
                session.active_run_id = None
                session.active_thread = None
            if not session.connections:
                self._sessions.pop(session_id, None)

    def active_run_id(self, session_id: str) -> str | None:
        """Return the active run identifier for one session workspace."""
        with self._lock:
            session = self._sessions.get(session_id)
            return session.active_run_id if session is not None else None

    def replace_baseline(
        self,
        session_id: str,
        genome: ArchitectureGenome,
    ) -> tuple[ArchitectureGenome, str]:
        """Replace only one session's baseline when no run is active."""
        with self._lock:
            session = self._get_or_create_session_locked(session_id)
            if session.active_run_id is not None:
                raise RuntimeError("cannot replace topology while an optimization run is active")
            session.baseline_genome = genome.model_copy(deep=True)
            return session.baseline_genome.model_copy(deep=True), session.instance_id

    async def _broadcast(
        self,
        session_id: str,
        session_instance_id: str,
        event: dict[str, Any],
    ) -> None:
        """Broadcast one event to connections in the matching workspace only."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.instance_id != session_instance_id:
                return
            connections = tuple(session.connections)
        if not connections:
            return

        outcomes = await asyncio.gather(
            *(connection.send_json(event) for connection in connections),
            return_exceptions=True,
        )
        stale_connections = [
            connection
            for connection, outcome in zip(connections, outcomes, strict=True)
            if isinstance(outcome, Exception)
        ]
        if stale_connections:
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None or session.instance_id != session_instance_id:
                    return
                for connection in stale_connections:
                    session.connections.discard(connection)
                if not session.connections:
                    self._sessions.pop(session_id, None)

    def _get_or_create_session_locked(self, session_id: str) -> DashboardSession:
        """Return a workspace while the state manager lock is held."""
        session = self._sessions.get(session_id)
        if session is None:
            session = DashboardSession(
                session_id=session_id,
                instance_id=uuid4().hex,
                baseline_genome=self._baseline_template.model_copy(deep=True),
            )
            self._sessions[session_id] = session
        return session

    @staticmethod
    def _consume_broadcast_result(future: Future[Any]) -> None:
        """Prevent background broadcast failures from becoming unhandled futures."""
        try:
            future.result()
        except Exception:
            LOGGER.exception("Unable to broadcast dashboard event")


def create_app() -> FastAPI:
    """Build the FastAPI application used by the EvoArch dashboard."""
    application = FastAPI(
        title="EvoArch Control Center",
        version="1.0.0",
        description="Live evolutionary microservice topology optimization.",
    )
    dashboard_state = DashboardState(_build_baseline_genome())
    application.state.dashboard_state = dashboard_state
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @application.get("/", response_class=FileResponse)
    async def dashboard_page(request: Request) -> FileResponse:
        """Serve the standalone split-screen visualization interface."""
        existing_session_id = request.cookies.get(SESSION_COOKIE_NAME)
        session_id = _normalize_session_id(existing_session_id)
        response = FileResponse(TEMPLATE_PATH, media_type="text/html")
        if session_id != existing_session_id:
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_id,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https",
            )
        return response

    @application.get("/api/status")
    async def optimization_status(request: Request) -> dict[str, str | None]:
        """Expose the requesting browser session's active optimization worker."""
        return {"active_run_id": dashboard_state.active_run_id(_http_session_id(request))}

    @application.post("/api/verify-access")
    async def verify_access(
        x_access_code: str | None = Header(None),
    ) -> dict[str, str]:
        """Verify that a dashboard user holds the current access code."""
        _require_access_code(x_access_code)
        return {"status": "authenticated"}

    @application.post(
        "/api/run-optimization",
        response_model=OptimizationRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    @limiter.limit("12/minute", key_func=lambda request: "global")
    @limiter.limit("5/minute")
    async def run_optimization(
        request: Request,
        optimization_request: OptimizationRunRequest,
        x_access_code: str | None = Header(None),
    ) -> OptimizationRunResponse:
        """Translate intent, then schedule its constrained optimization worker."""
        _require_access_code(x_access_code)
        session_id = _http_session_id(request)
        try:
            intent_weights = await _translate_intent_to_weights(
                optimization_request.user_prompt,
                chaos_mode=optimization_request.chaos_mode,
            )
        except (AIAgentError, ValidationError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Unable to translate optimization intent: {error}",
            ) from error

        try:
            run_id = dashboard_state.start_optimization(
                session_id,
                optimization_request,
                intent_weights,
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        return OptimizationRunResponse(run_id=run_id, status="started")

    @application.post("/api/upload-topology", response_model=TopologyUploadResponse)
    async def upload_topology(
        request: Request,
        file: UploadFile = File(...),
    ) -> TopologyUploadResponse:
        """Parse an uploaded Compose or Kubernetes YAML file into the baseline DNA."""
        session_id = _http_session_id(request)
        filename = file.filename or "topology.yaml"
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_TOPOLOGY_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Upload a .yaml or .yml Docker Compose or Kubernetes file",
            )

        try:
            raw_content = await file.read(MAX_TOPOLOGY_UPLOAD_BYTES + 1)
            if len(raw_content) > MAX_TOPOLOGY_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Topology uploads must be 1 MB or smaller",
                )
            try:
                file_content = raw_content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Topology file must be UTF-8 encoded text",
                ) from error
            _validate_topology_yaml(file_content)

            try:
                genome = await _parse_infrastructure_to_genome(file_content)
                probe_result = TrafficSimulator().simulate_load(
                    genome,
                    INGESTION_PROBE_QPS,
                )
                probe_record = build_pareto_fitness_records(
                    [genome],
                    [probe_result],
                )[0]
            except (AIAgentError, ValidationError, TypeError, ValueError) as error:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unable to parse a valid acyclic EvoArch topology: {error}",
                ) from error

            try:
                baseline_genome, session_instance_id = dashboard_state.replace_baseline(
                    session_id,
                    genome,
                )
            except RuntimeError as error:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(error),
                ) from error

            dashboard_state.publish(
                session_id,
                {
                    "event": "topology_ingested",
                    "run_id": None,
                    "timestamp": _timestamp(),
                    "generation": None,
                    "layout": _serialize_layout(baseline_genome, probe_record, set()),
                    "metrics": _serialize_metrics(probe_record),
                    "logs": [
                        {
                            "level": "success",
                            "message": (
                                f"Ingested {filename}: {len(baseline_genome.services)} "
                                f"services and {len(baseline_genome.edges)} dependencies. "
                                f"Completed a {INGESTION_PROBE_QPS:.1f} QPS mathematical "
                                "baseline probe."
                            ),
                        }
                    ],
                },
                session_instance_id=session_instance_id,
            )
            return TopologyUploadResponse(
                status="ingested",
                source_file=filename,
                service_count=len(baseline_genome.services),
                edge_count=len(baseline_genome.edges),
            )
        finally:
            await file.close()

    @application.websocket("/ws/stream")
    async def stream_optimization(websocket: WebSocket) -> None:
        """Stream optimization events and retain the socket until it disconnects."""
        connected = False
        session_id = _websocket_session_id(websocket)
        try:
            await dashboard_state.connect(session_id, websocket)
            connected = True
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if connected:
                dashboard_state.disconnect(session_id, websocket)

    return application


async def _translate_intent_to_weights(
    user_prompt: str,
    *,
    chaos_mode: bool = False,
) -> IntentWeights:
    """Call the AI control plane and revalidate its boundary-safe output."""
    agent = EvoArchAIAgent()
    try:
        translated_weights = await agent.translate_intent_to_weights(
            user_prompt,
            chaos_mode=chaos_mode,
        )
        return IntentWeights.model_validate(translated_weights)
    finally:
        await _close_agent(agent)


async def _parse_infrastructure_to_genome(file_content: str) -> ArchitectureGenome:
    """Parse an uploaded Compose or Kubernetes document without an AI request."""
    return parse_infrastructure_to_genome(file_content)


def _validate_topology_yaml(file_content: str) -> None:
    """Reject empty, malformed, or non-structural YAML before local extraction."""
    if not file_content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Topology file cannot be empty",
        )
    try:
        documents = [document for document in yaml.safe_load_all(file_content) if document is not None]
    except yaml.YAMLError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Topology file is not valid YAML",
        ) from error
    if not documents or any(not isinstance(document, (dict, list)) for document in documents):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Topology YAML must contain at least one mapping or list document",
        )


async def _generate_deployment_package(
    optimized_genome: dict[str, Any],
    *,
    chaos_mode: bool = False,
) -> dict[str, str]:
    """Synthesize validated Kubernetes artifacts in the worker's event loop."""
    agent = EvoArchAIAgent()
    try:
        package = await agent.generate_deployment_package(
            optimized_genome,
            chaos_mode=chaos_mode,
        )
        adr_markdown = package.get("adr_markdown")
        iac_code = package.get("iac_code")
        if not isinstance(adr_markdown, str) or not isinstance(iac_code, str):
            raise AIAgentError("deployment package is missing textual artifacts")
        return {"adr_markdown": adr_markdown, "iac_code": iac_code}
    finally:
        await _close_agent(agent)


async def _close_agent(agent: EvoArchAIAgent) -> None:
    """Close a request-scoped SDK client without masking the primary failure."""
    try:
        await agent.close()
    except Exception:
        LOGGER.warning("Unable to close EvoArch AI agent cleanly", exc_info=True)


def _intent_translated_event(
    run_id: str,
    intent_weights: IntentWeights,
    *,
    chaos_mode: bool,
) -> dict[str, Any]:
    """Create the audit event emitted after intent becomes math constraints."""
    return {
        "event": "intent_translated",
        "run_id": run_id,
        "timestamp": _timestamp(),
        "generation": None,
        "chaos_mode": chaos_mode,
        "layout": [],
        "metrics": None,
        "intent_weights": intent_weights.model_dump(),
        "logs": [
            {
                "level": "info",
                "message": (
                    "AI control plane translated intent: "
                    f"latency_weight={intent_weights.latency_weight:.2f}; "
                    f"cost_weight={intent_weights.cost_weight:.2f}; "
                    f"max_replicas_cap={intent_weights.max_replicas_cap}; "
                    "load_intensity_multiplier="
                    f"{intent_weights.load_intensity_multiplier:.2f}"
                    f"; chaos_mode={'enabled' if chaos_mode else 'disabled'}."
                ),
            }
        ],
    }


def _run_optimization(
    dashboard_state: DashboardState,
    session_id: str,
    session_instance_id: str,
    run_id: str,
    request: OptimizationRunRequest,
    intent_weights: IntentWeights,
    baseline_genome: ArchitectureGenome,
) -> None:
    """Execute generations, publishing one visualization event per completion."""
    try:
        simulator = TrafficSimulator()
        effective_baseline_qps = (
            request.baseline_qps * intent_weights.load_intensity_multiplier
        )
        engine = EvolutionEngine(
            simulator,
            effective_baseline_qps,
            elite_count=max(1, request.population_size // 10),
            tournament_size=3,
            mutation_rate=0.80,
            max_workers=min(32, request.population_size),
            random_seed=request.random_seed,
            latency_weight=intent_weights.latency_weight,
            cost_weight=intent_weights.cost_weight,
            max_replicas_cap=intent_weights.max_replicas_cap,
            chaos_mode=request.chaos_mode,
        )
        baseline_genome = _cap_genome_replicas(
            baseline_genome,
            intent_weights.max_replicas_cap,
        )
        population = _build_initial_population(
            baseline_genome,
            request.population_size,
            engine,
        )
        dashboard_state.publish(
            session_id,
            {
                "event": "run_started",
                "run_id": run_id,
                "timestamp": _timestamp(),
                "generation": 0,
                "chaos_mode": request.chaos_mode,
                "layout": _serialize_layout(baseline_genome, None, set()),
                "metrics": None,
                "logs": [
                    {
                        "level": "info",
                        "message": (
                            f"Run {run_id[:8]} started with {request.population_size} "
                            f"architectures at {effective_baseline_qps:.1f} QPS "
                            "after AI load scaling."
                        ),
                    },
                    *(
                        [
                            {
                                "level": "warning",
                                "message": (
                                    "Chaos Mode armed: each generation removes one "
                                    "replica from a shared random 10–20% service sample."
                                ),
                            }
                        ]
                        if request.chaos_mode
                        else []
                    ),
                ],
            },
            session_instance_id=session_instance_id,
        )

        previous_best: ArchitectureGenome | None = None
        previous_record: FitnessRecord | None = None
        best_genome: ArchitectureGenome | None = None
        best_record: FitnessRecord | None = None
        for generation in range(1, request.generation_count + 1):
            evaluated_population = population
            population = engine.run_generation(evaluated_population)
            best_index = _best_record_index(engine.last_fitness)
            best_genome = evaluated_population[best_index]
            best_record = engine.last_fitness[best_index]
            mutated_services = _changed_services(previous_best, best_genome)
            logs = _generation_logs(
                generation,
                previous_best,
                best_genome,
                previous_record,
                best_record,
            )

            dashboard_state.publish(
                session_id,
                {
                    "event": "generation_complete",
                    "run_id": run_id,
                    "timestamp": _timestamp(),
                    "generation": generation,
                    "chaos_mode": request.chaos_mode,
                    "layout": _serialize_layout(
                        best_genome,
                        best_record,
                        mutated_services,
                    ),
                    "metrics": _serialize_metrics(best_record),
                    "logs": logs,
                },
                session_instance_id=session_instance_id,
            )
            previous_best = best_genome.model_copy(deep=True)
            previous_record = best_record.copy()

            if request.generation_delay_ms > 0:
                sleep(request.generation_delay_ms / 1_000.0)

        if best_genome is None or best_record is None:
            raise RuntimeError("optimization completed without an evaluated genome")

        if not _is_feasible_architecture(best_record):
            infeasibility = _infeasibility_details(best_record)
            dashboard_state.publish(
                session_id,
                {
                    "event": "run_infeasible",
                    "run_id": run_id,
                    "timestamp": _timestamp(),
                    "generation": request.generation_count,
                    "chaos_mode": request.chaos_mode,
                    "layout": _serialize_layout(best_genome, best_record, set()),
                    "metrics": _serialize_metrics(best_record),
                    "infeasibility": infeasibility,
                    "logs": [
                        {
                            "level": "error",
                            "message": (
                                "No queue-stable architecture was found within EvoArch's "
                                "resource bounds; deployment synthesis was intentionally "
                                "withheld."
                            ),
                        },
                        {
                            "level": "warning",
                            "message": _infeasibility_log_message(infeasibility),
                        },
                    ],
                },
                session_instance_id=session_instance_id,
            )
            return

        dashboard_state.publish(
            session_id,
            {
                "event": "artifact_synthesis_started",
                "run_id": run_id,
                "timestamp": _timestamp(),
                "generation": request.generation_count,
                "chaos_mode": request.chaos_mode,
                "layout": _serialize_layout(best_genome, best_record, set()),
                "metrics": _serialize_metrics(best_record),
                "logs": [
                    {
                        "level": "info",
                        "message": (
                            "Evolution complete. Racing AI deployment models for the "
                            "first validated ADR and Kubernetes manifest package."
                        ),
                    }
                ],
            },
            session_instance_id=session_instance_id,
        )
        deployment_package = asyncio.run(
            _generate_deployment_package(
                best_genome.model_dump(mode="json"),
                chaos_mode=request.chaos_mode,
            )
        )

        dashboard_state.publish(
            session_id,
            {
                "event": "run_complete",
                "run_id": run_id,
                "timestamp": _timestamp(),
                "generation": request.generation_count,
                "chaos_mode": request.chaos_mode,
                "layout": _serialize_layout(best_genome, best_record, set()),
                "metrics": _serialize_metrics(best_record),
                "adr_markdown": deployment_package["adr_markdown"],
                "iac_code": deployment_package["iac_code"],
                "target_format": "kubernetes",
                "logs": [
                    {
                        "level": "success",
                        "message": (
                            f"Run {run_id[:8]} completed after "
                            f"{request.generation_count} generations; "
                            "validated deployment artifacts are ready."
                        ),
                    }
                ],
            },
            session_instance_id=session_instance_id,
        )
    except Exception as error:
        LOGGER.exception("Optimization dashboard worker failed")
        dashboard_state.publish(
            session_id,
            {
                "event": "run_failed",
                "run_id": run_id,
                "timestamp": _timestamp(),
                "generation": None,
                "layout": [],
                "metrics": None,
                "logs": [
                    {
                        "level": "error",
                        "message": f"Optimization failed: {type(error).__name__}: {error}",
                    }
                ],
            },
            session_instance_id=session_instance_id,
        )
    finally:
        dashboard_state.finish_optimization(
            session_id,
            session_instance_id,
            run_id,
        )


def _build_baseline_genome() -> ArchitectureGenome:
    """Create the intentionally constrained topology shown when a run begins."""
    services = {
        "api-gateway": ServiceGene(
            service_name="api-gateway",
            replicas=2,
            cpu_limit=1.5,
            mem_limit_mb=1024,
            routing_algorithm="round_robin",
        ),
        "auth-service": ServiceGene(
            service_name="auth-service",
            replicas=2,
            cpu_limit=1.0,
            mem_limit_mb=512,
            routing_algorithm="least_connections",
        ),
        "catalog-service": ServiceGene(
            service_name="catalog-service",
            replicas=2,
            cpu_limit=1.0,
            mem_limit_mb=1024,
            routing_algorithm="round_robin",
        ),
        "cart-service": ServiceGene(
            service_name="cart-service",
            replicas=2,
            cpu_limit=1.0,
            mem_limit_mb=768,
            routing_algorithm="random",
        ),
        "payment-service": ServiceGene(
            service_name="payment-service",
            replicas=1,
            cpu_limit=0.75,
            mem_limit_mb=512,
            routing_algorithm="least_connections",
        ),
        "inventory-service": ServiceGene(
            service_name="inventory-service",
            replicas=2,
            cpu_limit=0.75,
            mem_limit_mb=512,
            routing_algorithm="round_robin",
        ),
        "notification-service": ServiceGene(
            service_name="notification-service",
            replicas=1,
            cpu_limit=0.5,
            mem_limit_mb=256,
            routing_algorithm="random",
        ),
    }
    return ArchitectureGenome(
        services=services,
        edges=[
            EdgeGene(
                source="api-gateway",
                target="auth-service",
                base_latency_ms=2.0,
            ),
            EdgeGene(
                source="api-gateway",
                target="catalog-service",
                base_latency_ms=4.0,
            ),
            EdgeGene(
                source="api-gateway",
                target="cart-service",
                base_latency_ms=3.0,
            ),
            EdgeGene(
                source="cart-service",
                target="payment-service",
                base_latency_ms=6.0,
            ),
            EdgeGene(
                source="cart-service",
                target="inventory-service",
                base_latency_ms=5.0,
            ),
            EdgeGene(
                source="payment-service",
                target="notification-service",
                base_latency_ms=8.0,
            ),
        ],
    )


def _cap_genome_replicas(
    genome: ArchitectureGenome,
    max_replicas_cap: int,
) -> ArchitectureGenome:
    """Apply the intent-derived replica ceiling before the first evaluation."""
    if not 1 <= max_replicas_cap <= 20:
        raise ValueError("max_replicas_cap must be between 1 and 20")
    return ArchitectureGenome(
        services={
            service_name: ServiceGene(
                **{
                    **service.model_dump(),
                    "replicas": min(service.replicas, max_replicas_cap),
                }
            )
            for service_name, service in genome.services.items()
        },
        edges=[edge.model_copy(deep=True) for edge in genome.edges],
    )


def _build_initial_population(
    baseline_genome: ArchitectureGenome,
    population_size: int,
    engine: EvolutionEngine,
) -> list[ArchitectureGenome]:
    """Seed a varied first population while preserving the baseline topology."""
    strategies: tuple[MutationStrategy, ...] = (
        "scale_replicas",
        "adjust_resources",
        "toggle_routing",
        "break_bottleneck",
    )
    population = [baseline_genome.model_copy(deep=True)]
    for index in range(1, population_size):
        candidate = engine.mutate(
            baseline_genome,
            strategy=strategies[(index - 1) % len(strategies)],
        )
        if index >= len(strategies):
            candidate = engine.mutate(
                candidate,
                strategy=strategies[index % len(strategies)],
            )
        population.append(candidate)
    return population


def _best_record_index(records: Sequence[FitnessRecord]) -> int:
    """Select the dashboard representative with NSGA-II selection precedence."""
    if not records:
        raise ValueError("at least one fitness record is required")

    def record_key(index: int) -> tuple[int, float, float, float, int]:
        record = records[index]
        crowding_distance = record["crowding_distance"]
        normalized_crowding = (
            crowding_distance if isfinite(crowding_distance) else inf
        )
        latency = record["total_p99_latency_ms"]
        return (
            record["front_rank"],
            -normalized_crowding,
            -record["composite_fitness"],
            inf if latency is None else latency,
            record["genome_index"],
        )

    return min(range(len(records)), key=record_key)


def _is_feasible_architecture(record: FitnessRecord) -> bool:
    """Return whether a record is safe to materialize into deployment artifacts."""
    return (
        record["simulation_error"] is None
        and record["total_p99_latency_ms"] is not None
        and not record["queue_saturation"]
    )


def _infeasibility_details(record: FitnessRecord) -> dict[str, Any]:
    """Expose the most severe capacity shortfalls for dashboard diagnostics."""
    bottlenecks = sorted(
        (
            {
                "service_name": service_name,
                "arrival_rate_qps": round(details["arrival_rate_qps"], 1),
                "capacity_qps": round(details["capacity_qps"], 1),
                "excess_qps": round(details["excess_qps"], 1),
            }
            for service_name, details in record["queue_saturation"].items()
        ),
        key=lambda item: item["excess_qps"],
        reverse=True,
    )
    return {
        "saturated_service_count": len(record["queue_saturation"]),
        "simulation_error": record["simulation_error"],
        "bottlenecks": bottlenecks[:3],
    }


def _infeasibility_log_message(infeasibility: dict[str, Any]) -> str:
    """Describe the capacity ceiling without exposing a misleading success state."""
    bottlenecks = infeasibility["bottlenecks"]
    if not bottlenecks:
        return "The simulation did not return a valid finite latency result."

    summary = "; ".join(
        (
            f"{bottleneck['service_name']} receives "
            f"{bottleneck['arrival_rate_qps']:.1f} QPS but has "
            f"{bottleneck['capacity_qps']:.1f} QPS capacity"
        )
        for bottleneck in bottlenecks
    )
    return (
        f"{infeasibility['saturated_service_count']} service queues remain "
        f"saturated. Largest shortfalls: {summary}."
    )


def _serialize_layout(
    genome: ArchitectureGenome,
    fitness_record: FitnessRecord | None,
    mutated_services: set[str],
) -> list[dict[str, Any]]:
    """Convert a genome into Cytoscape elements with live queueing annotations."""
    service_metrics = (
        fitness_record["service_metrics"] if fitness_record is not None else {}
    )
    saturated_services = (
        set(fitness_record["queue_saturation"]) if fitness_record is not None else set()
    )
    elements: list[dict[str, Any]] = []

    for service_name, service in genome.services.items():
        metric = service_metrics.get(service_name)
        utilization = metric["utilization"] if metric is not None else 0.0
        arrival_rate = metric["arrival_rate_qps"] if metric is not None else 0.0
        utilization_ratio = _utilization_ratio(metric)
        chaos_failure_injected = (
            metric["chaos_failure_injected"] if metric is not None else False
        )
        if service_name in saturated_services:
            node_status = "saturated"
        elif chaos_failure_injected:
            node_status = "chaos-impacted"
        elif service_name in mutated_services:
            node_status = "mutating"
        else:
            node_status = "optimized"

        elements.append(
            {
                "group": "nodes",
                "data": {
                    "id": service_name,
                    "label": service_name,
                    "replicas": service.replicas,
                    "cpu_limit": service.cpu_limit,
                    "memory_mb": service.mem_limit_mb,
                    "routing_algorithm": service.routing_algorithm,
                    "cpu_load_percent": (
                        round(utilization * 100.0, 1)
                        if utilization is not None
                        else None
                    ),
                    "utilization_ratio": utilization_ratio,
                    "arrival_rate_qps": round(arrival_rate, 3),
                    "available_replicas": (
                        metric["available_replicas"]
                        if metric is not None
                        else service.replicas
                    ),
                    "chaos_failure_injected": chaos_failure_injected,
                    "node_size": round(
                        max(54.0, min(118.0, 54.0 + service.replicas * 5.0 + service.cpu_limit * 12.0)),
                        1,
                    ),
                    "status": node_status,
                },
                "classes": node_status,
            }
        )

    for edge_index, edge in enumerate(genome.edges):
        source_metric = service_metrics.get(edge.source)
        target_metric = service_metrics.get(edge.target)
        active = source_metric is not None and source_metric["arrival_rate_qps"] > 0.0
        dynamic_latency_ms = _dynamic_edge_latency_ms(edge.base_latency_ms, target_metric)
        edge_saturated = target_metric is not None and dynamic_latency_ms is None
        has_meaningful_queue_delay = (
            dynamic_latency_ms is not None
            and dynamic_latency_ms
            >= edge.base_latency_ms + EDGE_LABEL_QUEUE_DELAY_THRESHOLD_MS
        )
        elements.append(
            {
                "group": "edges",
                "data": {
                    "id": f"edge-{edge_index}-{edge.source}-{edge.target}",
                    "source": edge.source,
                    "target": edge.target,
                    "base_latency_ms": edge.base_latency_ms,
                    "dynamic_latency_ms": dynamic_latency_ms,
                    "show_latency_label": has_meaningful_queue_delay,
                    "edge_saturated": edge_saturated,
                    "active": active,
                },
                "classes": "active-path" if active else "idle-path",
            }
        )

    return elements


def _utilization_ratio(metric: ServiceLoadDetails | None) -> float:
    """Return a finite capacity ratio suitable for Cytoscape heatmap mapping."""
    if metric is None:
        return 0.0

    capacity_qps = metric["capacity_qps"]
    if capacity_qps <= 0.0:
        return 1.0
    return round(max(0.0, metric["arrival_rate_qps"] / capacity_qps), 4)


def _dynamic_edge_latency_ms(
    base_latency_ms: float,
    target_metric: ServiceLoadDetails | None,
) -> float | None:
    """Return network latency plus the target M/M/c queue's expected wait time.

    For a stable M/M/c queue, Erlang C gives ``W_q = P(wait) / (c * mu -
    lambda)``. The service metrics expose the equivalent capacity and arrival
    rates, allowing the dashboard to label every edge with its current live
    latency without rerunning the simulator. A saturated queue has no finite
    waiting time, so ``None`` intentionally reaches the UI as ``SAT``.
    """
    if target_metric is None:
        return round(base_latency_ms, 1)

    wait_probability = target_metric["erlang_c_wait_probability"]
    remaining_capacity_qps = (
        target_metric["capacity_qps"] - target_metric["arrival_rate_qps"]
    )
    if wait_probability is None or remaining_capacity_qps <= 0.0:
        return None

    queue_delay_ms = 1_000.0 * wait_probability / remaining_capacity_qps
    return round(base_latency_ms + queue_delay_ms, 1)


def _serialize_metrics(record: FitnessRecord) -> dict[str, Any]:
    """Prepare finite, JSON-safe aggregate metrics for frontend rendering."""
    return {
        "p99_latency_ms": record["total_p99_latency_ms"],
        "cost_hourly": round(record["total_cost_hourly"], 4),
        "front_rank": record["front_rank"],
        "composite_fitness": round(record["composite_fitness"], 6),
        "saturated_services": sorted(record["queue_saturation"]),
        "chaos_mode": record["chaos_mode"],
        "chaos_failed_services": list(record["chaos_failed_services"]),
    }


def _changed_services(
    previous: ArchitectureGenome | None,
    current: ArchitectureGenome,
) -> set[str]:
    """Identify nodes whose resource or routing genes changed since last display."""
    if previous is None:
        return set()

    changed = set(previous.services) ^ set(current.services)
    for service_name in set(previous.services) & set(current.services):
        if previous.services[service_name] != current.services[service_name]:
            changed.add(service_name)
    return changed


def _generation_logs(
    generation: int,
    previous_genome: ArchitectureGenome | None,
    current_genome: ArchitectureGenome,
    previous_record: FitnessRecord | None,
    current_record: FitnessRecord,
) -> list[dict[str, str]]:
    """Create concise, human-readable audit records for the live terminal."""
    if previous_genome is None:
        action = "Evaluated the seeded Pareto population"
    else:
        action = _describe_architecture_change(previous_genome, current_genome)

    latency_change = _latency_change_message(previous_record, current_record)
    logs = [
        {
            "level": "info",
            "message": f"Generation {generation}: {action}. {latency_change}",
        }
    ]
    if current_record["chaos_mode"]:
        logs.append(
            {
                "level": "warning",
                "message": (
                    "Chaos Mode removed one replica from: "
                    f"{', '.join(current_record['chaos_failed_services'])}."
                ),
            }
        )
    saturated_services = sorted(current_record["queue_saturation"])
    if saturated_services:
        logs.append(
            {
                "level": "warning",
                "message": (
                    "Erlang C queue saturation detected in "
                    f"{', '.join(saturated_services)}."
                ),
            }
        )
    else:
        logs.append(
            {
                "level": "success",
                "message": (
                    f"Pareto front {current_record['front_rank']} is queue-stable "
                    f"at ${current_record['total_cost_hourly']:.4f}/hour."
                ),
            }
        )
    return logs


def _describe_architecture_change(
    previous: ArchitectureGenome,
    current: ArchitectureGenome,
) -> str:
    """Describe the highest-signal gene difference between two genomes."""
    for service_name in current.services:
        if service_name not in previous.services:
            return f"Added {service_name} to the active topology"
        old_service = previous.services[service_name]
        new_service = current.services[service_name]
        if old_service.replicas != new_service.replicas:
            return f"Scaled {service_name} to {new_service.replicas} replicas"
        if old_service.cpu_limit != new_service.cpu_limit:
            return (
                f"Adjusted {service_name} CPU limit to {new_service.cpu_limit:.2f} cores"
            )
        if old_service.mem_limit_mb != new_service.mem_limit_mb:
            return f"Adjusted {service_name} memory to {new_service.mem_limit_mb} MB"
        if old_service.routing_algorithm != new_service.routing_algorithm:
            return (
                f"Switched {service_name} routing to {new_service.routing_algorithm}"
            )
    for service_name in previous.services:
        if service_name not in current.services:
            return f"Removed {service_name} from the active topology"
    return "Retained the leading architecture through elitism"


def _latency_change_message(
    previous: FitnessRecord | None,
    current: FitnessRecord,
) -> str:
    """Describe latency progress while handling saturated queues explicitly."""
    current_latency = current["total_p99_latency_ms"]
    if previous is None:
        if current_latency is None:
            return "P99 latency is unbounded while the queue remains saturated"
        return f"P99 latency is {current_latency:.2f} ms"

    previous_latency = previous["total_p99_latency_ms"]
    if previous_latency is None and current_latency is None:
        return "P99 latency remains unbounded while queue saturation persists"
    if previous_latency is None:
        return f"P99 latency recovered to {current_latency:.2f} ms"
    if current_latency is None:
        return "P99 latency became unbounded due to queue saturation"

    difference = previous_latency - current_latency
    if difference > 0.005:
        return f"P99 latency dropped by {difference:.2f} ms"
    if difference < -0.005:
        return f"P99 latency increased by {abs(difference):.2f} ms"
    return f"P99 latency held at {current_latency:.2f} ms"


def _timestamp() -> str:
    """Return a UTC timestamp for an event payload."""
    return datetime.now(timezone.utc).isoformat()


app = create_app()
