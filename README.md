# EvoArch

> **An AI-assisted, mathematics-first reliability and cost optimizer for microservice architectures.**
>
> EvoArch is a design-time decision-support tool. It turns an infrastructure topology and a developer objective into a queueing-theory simulation, a Pareto-optimized architecture genome, and validated deployment guidance.

## Built End-to-End with Codex & GPT-5.6

EvoArch's architecture, mathematical models, and streaming backend were designed and coded **100% using Codex**. Codex built the system's core mathematical, evolutionary, parsing, API, and visualization layers—not merely the interface.

| Codex-built subsystem | What Codex implemented |
| --- | --- |
| Queueing engine | M/M/c service modeling, Erlang C waiting probabilities, P99 response-time estimation, saturation detection, and cost calculations. |
| Evolution engine | NSGA-II Pareto ranking, crowding distance, tournament selection, elitism, resource crossover, and bottleneck-targeted mutations. |
| Deterministic parser | Docker Compose and Kubernetes YAML extraction into a strict Pydantic v2 `ArchitectureGenome`, with **zero LLM token use** during dashboard ingestion. |
| Streaming control plane | Async FastAPI routes, WebSocket event streaming, per-session dashboard isolation, access gating, and rate limiting. |
| Operator dashboard | Cytoscape.js topology rendering, queue-saturation heatmaps, dynamic edge latency, audit logs, and validated IaC output. |

The AI Control Plane can use a direct **OpenAI/GPT-5.6** configuration (`gpt-5.6-terra` is the repository default) or Gemini through its OpenAI-compatible endpoint. Gemini is selected automatically when `GEMINI_API_KEY` is configured; set `EVOARCH_AI_PROVIDER=openai` to explicitly use the direct OpenAI path.

> **Important:** The LLM does not perform the mathematical optimization. It translates bounded human intent into optimizer controls and synthesizes artifacts only after the deterministic simulation and evolutionary engine select a feasible architecture.

## Capabilities

| Feature | What it does |
| --- | --- |
| Infrastructure DNA | Represents services, resource limits, replicas, routing, and directed dependencies as validated Pydantic v2 genes. |
| Deterministic topology ingestion | Parses Docker Compose and Kubernetes YAML locally into an acyclic `ArchitectureGenome`; dashboard uploads do not call an LLM. |
| M/M/c simulation | Propagates dependency traffic, computes utilization and Erlang C wait behavior, and estimates critical-path P99 latency. |
| Transparent cost model | Calculates hourly compute spend from configured CPU, memory, and replica allocations. |
| NSGA-II optimization | Concurrently evaluates a population, applies non-dominated sorting and crowding distance, and preserves Pareto diversity. |
| Targeted evolution | Uses elitism, tournament selection, resource crossover, routing changes, replica scaling, and bottleneck-breaking mutations. |
| Chaos Mode | Removes one replica from a shared 10–20% random service sample each generation to reward fault-tolerant topologies. |
| AI Control Plane | Converts natural language into bounded fitness controls, then produces an ADR and validated IaC for the selected feasible genome. |
| Live operator view | Streams generation events to a Cytoscape.js dashboard with service utilization, saturation state, and dynamic path latency. |
| Deployment safeguards | Includes access-code validation, per-IP/global request limits, and session-isolated dashboard workspaces. |

## Architecture

```mermaid
flowchart LR
    User["Platform Engineer"] --> UI["EvoArch Dashboard\nFastAPI + Cytoscape.js"]
    UI --> Upload["Docker Compose / Kubernetes YAML"]
    Upload --> Parser["Deterministic YAML Parser\nZero LLM Tokens"]
    Parser --> Genome["Pydantic v2\nArchitectureGenome"]

    User --> Intent["Natural-language objective"]
    Intent --> Agent["AI Control Plane\nGPT-5.6 / configured provider"]
    Agent --> Weights["Validated IntentWeights\nlatency · cost · replica cap · load"]

    Genome --> Engine["EvolutionEngine\nNSGA-II + mutation + crossover"]
    Weights --> Engine
    Engine <--> Simulator["TrafficSimulator\nM/M/c + Erlang C"]
    Simulator --> Pareto["Feasibility, P99, Cost\nPareto Fitness"]
    Pareto --> Engine

    Engine --> Events["WebSocket generation events"]
    Events --> UI
    Engine --> Best["Best feasible ArchitectureGenome"]
    Best --> Agent
    Agent --> Validate["ADR + Kubernetes/Terraform\nstrict genome validation"]
    Validate --> Output["Validated deployment package"]
    Output --> UI
```

### Core workflow

1. Upload a Docker Compose or Kubernetes manifest.
2. EvoArch deterministically parses and validates the topology as an `ArchitectureGenome`.
3. Enter a deployment objective such as: `Minimize checkout latency while preserving queue stability at 2x traffic.`
4. The AI Control Plane converts the objective into strict optimizer controls.
5. The evolutionary engine evaluates candidate topologies concurrently against the M/M/c simulation.
6. The dashboard streams Pareto generations, utilization, saturation, path latency, and audit events over WebSockets.
7. If a queue-stable design is found, the AI Control Plane generates an ADR and IaC package that EvoArch validates against the final genome.

> EvoArch intentionally withholds deployment synthesis when all candidates remain queue-saturated within the configured resource bounds. A plausible-looking YAML artifact is never treated as a valid result.

## Quick Start

### Prerequisites

- Python **3.10+**
- An OpenAI API key for direct GPT-5.6 use, or a Gemini API key for the default auto-selected Gemini path
- A supported Docker Compose or Kubernetes YAML file to optimize

### 1. Create a virtual environment

```bash
cd /path/to/EvoArch
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure the control plane

```bash
cp .env.example .env
```

For a direct OpenAI/GPT-5.6 deployment, configure:

```dotenv
OPENAI_API_KEY=sk-your-key
EVOARCH_AI_PROVIDER=openai
EVOARCH_OPENAI_MODEL=gpt-5.6-terra
ACCESS_CODE=replace-with-a-private-dashboard-code
```

> The configured model route must be available to your OpenAI account. Do not commit `.env`; it is ignored by Git.

### 4. Start EvoArch

```bash
uvicorn evoarch.visualization.dashboard:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000), enter the configured Access DNA, ingest a YAML topology, and trigger an optimization run.

## Mathematical Model

EvoArch uses an analytic queueing model to compare architecture candidates under the same workload. It is intentionally transparent and deterministic.

### Traffic propagation

The configured `baseline_qps` is external workload at root services. Each declared dependency propagates its source service's request rate to the target, modeling a synchronous call per dependency. Arrival rates from multiple upstream services accumulate.

The topology must be a directed acyclic graph. This lets EvoArch propagate traffic deterministically in topological order and calculate a critical path.

### M/M/c and Erlang C

For each service:

- `λ` is the propagated arrival rate in requests per second.
- `μ` is the per-replica service rate, derived from CPU allocation.
- `c` is the number of available replicas.
- Total capacity is `c × μ`.
- Utilization is `ρ = λ / (c × μ)`.

When `ρ ≥ 1`, the queue has no finite steady-state P99 latency. EvoArch marks it saturated and treats the architecture as infeasible for deployment synthesis.

For stable queues, Erlang C determines the probability an arrival waits. EvoArch then solves the response-time distribution numerically for the P99, combining service time and queue wait behavior.

### Transparent cost model

The hourly compute estimate is deliberately simple and inspectable:

```text
hourly_cost = Σ replicas × ((cpu_limit × $0.04) + ((memory_mib / 1024) × $0.01))
```

This is a comparative planning model, not a cloud-provider invoice. Use provider pricing, reserved capacity, storage, network egress, and production measurements for final financial decisions.

## Evolutionary Optimizer & Chaos Mode

EvoArch treats a topology as a genome:

| Gene | Searchable properties |
| --- | --- |
| `ServiceGene` | replicas, CPU limit, memory limit, and routing algorithm |
| `EdgeGene` | directed service dependency and base network latency |
| `ArchitectureGenome` | the complete service map and dependency graph |

Each generation is evaluated concurrently. NSGA-II assigns Pareto fronts by minimizing latency and hourly cost, then uses crowding distance to retain diverse trade-offs. The engine applies:

- **Elitism** to preserve leading candidates.
- **Tournament selection** based on Pareto rank and crowding distance.
- **Crossover** to blend parent resource allocations.
- **Discrete mutations** for replica scaling, CPU/memory adjustments, routing changes, and queue-bottleneck remediation.

### Chaos Mode

Chaos Mode samples a shared random **10–20%** of services for a generation and removes one replica from each sampled service. A one-replica service therefore becomes unavailable for that scenario.

Candidates are evaluated against the same sampled failure scenario within the generation. This prevents luck-based comparisons and rewards architectures with real capacity headroom and redundancy.

> Chaos Mode does not claim to replace production chaos engineering. It is a mathematical resilience signal used during evolutionary search.

## AI Control Plane

The AI Control Plane has two narrow, validated responsibilities.

### 1. Intent translation

Given an objective such as:

```text
Prepare checkout for 3x seasonal traffic. Prioritize P99 latency and queue stability over cloud spend.
```

the configured model returns an `IntentWeights` object:

```json
{
  "latency_weight": 0.8,
  "cost_weight": 0.2,
  "max_replicas_cap": 20,
  "load_intensity_multiplier": 3.0
}
```

EvoArch validates the result before use:

- latency and cost weights must sum to exactly `1.0`
- weights remain within `[0.0, 1.0]`
- replica caps remain within `[1, 20]`
- workload multipliers remain within `[0.1, 10.0]`

### 2. Deployment synthesis and validation

Only after the evolutionary engine finds a feasible candidate, the model generates:

- a concise Markdown **Architectural Decision Record (ADR)**
- Kubernetes manifests or Terraform output matching the selected topology

EvoArch validates generated IaC against the mathematical genome before it reaches the dashboard. Validation checks include exact service naming, requested resource allocations, replica counts, and required artifact structure. Invalid model output is corrected or rejected; it is not silently deployed.

## Configuration

Copy `.env.example` to `.env` and configure the provider you intend to use.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Direct OpenAI API credential. Required when `EVOARCH_AI_PROVIDER=openai`. |
| `EVOARCH_AI_PROVIDER` | Optional provider override: `openai` or `gemini`. When blank, Gemini is selected if `GEMINI_API_KEY` exists; otherwise OpenAI is used. |
| `EVOARCH_OPENAI_MODEL` | Direct OpenAI model route. The repository default is `gpt-5.6-terra`. |
| `GEMINI_API_KEY` | Gemini credential used through Google's OpenAI-compatible endpoint. |
| `EVOARCH_GEMINI_MODEL` | Primary Gemini model override. |
| `EVOARCH_GEMINI_INTENT_FALLBACK_MODEL` | Gemini fallback used when intent schema validation fails. |
| `EVOARCH_GEMINI_DEPLOYMENT_FALLBACK_MODEL` | Gemini fallback used after deployment-package validation fails. |
| `EVOARCH_GEMINI_DEPLOYMENT_MODELS` | Optional comma-separated deployment models to race for the first valid artifact. |
| `ACCESS_CODE` | Access DNA required by the dashboard gate and protected optimization endpoint. |

## Security & Operational Safeguards

- Dashboard access is protected by `ACCESS_CODE` validation.
- Optimization requests are rate limited to **5/minute per IP** and **12/minute globally**.
- Browser sessions receive isolated topology, run, graph, event-history, and WebSocket workspace state.
- A disconnected session's workspace is removed from memory to avoid retaining stale Render process state.
- API keys remain server-side in `.env`; the browser never receives model-provider credentials.

## Project Layout

```text
evoarch/
├── api/
│   ├── ai_agent.py                # Provider-backed intent and artifact control plane
│   └── infrastructure_parser.py   # Deterministic Compose/Kubernetes parser
├── engine/
│   └── evolution.py               # Parallel evolutionary loop and mutations
├── models/
│   └── genome.py                  # Pydantic v2 architecture DNA
├── optimizer/
│   └── fitness.py                 # Pareto ranking and crowding distance
├── simulation/
│   └── traffic.py                 # M/M/c, Erlang C, P99, and cost modeling
└── visualization/
    ├── dashboard.py               # Monolithic FastAPI control plane and WebSockets
    └── templates/index.html        # Cytoscape.js operator dashboard
```

## Scope and Assumptions

EvoArch is intentionally a **design-time optimizer**. Its M/M/c model, request propagation, and cost equation are useful for comparative architectural reasoning but do not replace:

- production load testing
- real distributed tracing and observability data
- cloud-provider pricing analysis
- security review, change management, or deployment approval
- fault-injection experiments against live systems

Use EvoArch to generate a mathematically explainable candidate architecture, then validate it with production-grade engineering processes.

## License

No license is currently declared for this repository.
