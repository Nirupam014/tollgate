# Tollgate — Agent Token-Risk Prevention Platform

**Engineering Design Document**
Version 0.2 · Status: Pre-deploy core implemented (open source); platform planes are forward-looking · Audience: Engineering

> **What's built today vs. the platform vision.** The **pre-deploy control plane**
> described here — parsing, prediction, simulation, detectors, the strict agentic
> linter, scoring/gate, substitution, remediation, reporters, and the
> CLI/Action/CI integrations — is **implemented and shipping as the open-source
> `tollgate` package** (plus a labeled validation suite and a field-study harness).
> It is deterministic, makes **no LLM calls**, and **never executes the code it
> scans**. The multi-tenant SaaS planes (hosted API gateway, event bus, OLTP/OLAP
> stores, telemetry recalibration, hosted GitHub App, exec dashboard) remain the
> **target architecture**, not yet built. Sections below mark which is which.
>
> Tollgate measures and gates on **tokens** (consumption risk), not dollars; price
> is only ever a secondary, derived view. This is an agentic-risk gate — a linter /
> SAST for agent workflows — first, and a cost tool second.

---

## 0. Document map

1. Problem & product thesis
2. System design overview
3. Design principles & non-functional requirements
4. Microservice architecture *(platform vision)*
5. Event-driven ingestion & event model *(platform vision)*
6. Conceptual data model *(platform vision)*
7. Database schema (SQL DDL) *(platform vision)*
8. API contracts (OpenAPI 3.1) *(platform vision)*
9. Core engines — parsing, prediction, simulation, detection, **strict agentic lint**, scoring, substitution, remediation, **self-healing outputs** *(built)*
10. GitHub / CI integration *(built: Action + CLI + CI templates; hosted App is vision)*
11. Multi-tenancy, security & isolation *(platform vision)*
12. Observability & SLOs *(platform vision)*
13. Validation, field study & the feedback loop *(built)*
14. MVP roadmap & capability coverage

---

## 1. Problem & product thesis

Agent workflows fail in production in structural, repeatable ways: an unbounded loop runs 40,000 times, a context window balloons to 180K tokens per request, an agent delegates to itself without a cap, a model is called with no `max_tokens`. The dominant tools today are **reporting/observability** tools — they tell you what already happened. By then the tokens are spent and the incident already shipped.

**Tollgate is a prevention tool.** It moves the point of control *left* — from the incident/invoice to the pull request. Before code ships, it parses the agent/workflow definition, predicts token consumption, simulates usage under realistic traffic, flags the structural failure modes that cause runaway token use (context explosion, recursive/delegation loops, missing iteration caps, uncapped generation, unbounded fan-out), recommends cheaper model substitutions, and produces a deployment risk score that can **block a PR**.

The mental model: **a strict code reviewer / static analyzer (a "SAST/linter") specifically for agentic risk** — it flags only agentic failure modes, not generic code smells, and it does so deterministically without ever running your code.

Three planes:

- **Pre-deploy (the core, built):** deterministic static + simulated analysis gating PRs, with a tamper-evident, re-derivable verdict.
- **Policy plane:** evaluation of token ceilings, model allowlists, context limits, and loop guards against proposed (and, in the platform vision, live) workloads.
- **Telemetry plane (platform vision):** event-driven ingestion of production usage to recalibrate predictions, detect drift, and feed forecasts.

---

## 2. System design overview

```
                          ┌───────────────────────────────────────────────┐
                          │                  Clients                       │
                          │  GitHub App · Web Console · CLI · CI plugins    │
                          └───────────────┬───────────────────────────────┘
                                          │  HTTPS / Webhooks
                                  ┌───────▼────────┐
                                  │   API Gateway   │  authn, tenant routing,
                                  │  + Edge Authz   │  rate limit, quota
                                  └───────┬────────┘
                ┌─────────────────────────┼──────────────────────────────┐
                │                         │                              │
        ┌───────▼───────┐        ┌────────▼────────┐            ┌─────────▼────────┐
        │  Analysis API │        │   Policy API     │            │  Telemetry API   │
        │  (sync/async) │        │ (real-time eval) │            │  (ingest, OTLP)  │
        └───────┬───────┘        └────────┬────────┘            └─────────┬────────┘
                │                          │                              │
                │   publish jobs / events  │  read policies               │  raw spans/events
                ▼                          ▼                              ▼
        ┌───────────────────────────── Event Bus (Kafka) ──────────────────────────┐
        │  topics: ingestion.raw · analysis.requested · analysis.completed ·        │
        │  policy.violation · telemetry.usage · forecast.updated · pr.checked       │
        └───────┬───────────┬───────────┬───────────┬───────────┬──────────────────┘
                │           │           │           │           │
        ┌───────▼──┐ ┌──────▼─────┐ ┌───▼──────┐ ┌──▼───────┐ ┌─▼──────────┐
        │ Workflow │ │ Prediction │ │ Cost Sim │ │ Risk     │ │ Telemetry  │
        │ Parser   │ │ Engine     │ │ Engine   │ │ Detectors│ │ Processor  │
        └───────┬──┘ └──────┬─────┘ └───┬──────┘ └──┬───────┘ └─┬──────────┘
                │           │           │           │           │
                └───────────┴─────┬─────┴───────────┴───────────┘
                                  ▼
                        ┌───────────────────┐      ┌─────────────────────┐
                        │  Model Catalog &   │      │  OLTP (Postgres)    │
                        │  Pricing Service   │      │  per-tenant rows    │
                        └───────────────────┘      └─────────────────────┘
                                  │
                        ┌─────────▼─────────┐      ┌─────────────────────┐
                        │ Forecast Service  │      │  OLAP (ClickHouse)  │
                        │ (exec reporting)  │◀─────│  telemetry warehouse│
                        └───────────────────┘      └─────────────────────┘
```

Three logical planes map onto the same bus:

- **Pre-deploy pipeline:** `Workflow Parser → Prediction Engine → Cost Sim Engine → Risk Detectors → Risk Scorer → PR Check`.
- **Policy plane:** `Policy API → Policy Engine` evaluating rules in <50 ms against a request shape.
- **Telemetry plane:** `Telemetry API → Telemetry Processor → ClickHouse → Forecast Service`, with a feedback edge into the Prediction Engine for recalibration.

---

## 3. Design principles & non-functional requirements

**Principles**

1. **Prevention over reporting.** Every feature must answer "how does this stop a runaway *before* it happens?" Reporting exists only to recalibrate prediction.
2. **Deterministic, offline, and hands-off.** The analyzer is pure static analysis (Python `ast`) plus seeded Monte-Carlo math: **no LLM calls, no network, reproducible output**, and it **never imports or executes the code it scans**. Re-running on the same inputs yields the same verdict — which is what makes the self-healing fingerprint (§9.9) possible.
3. **Structural facts are separated from cost estimates.** Graph shape, loop edges, missing caps and context accumulation are computed deterministically and reported as facts; token counts are distributions (p50/p95/p99); dollar figures are only ever a derived, secondary view. The two are never blurred.
4. **Honest failure over silent wrong answers.** A file that can't be parsed into something analyzable is dropped, not scored as a confident PASS. Where a framework can be recognized but not fully graphed, the strict linter still gives a structural-only verdict (§9.8) rather than nothing.
5. **Agentic-only.** Every check is gated on an agentic signal (a known agent framework or a recognized LLM SDK call); the tool is silent on non-agentic code. It flags agentic risk, not generic code smells.
6. **Provider-agnostic core.** Model-specific behavior lives behind the Model Catalog adapter interface. Adding a provider ≠ touching the Risk Scorer.
7. **Fail safe, not fail open.** If confidence is low or analysis times out, surface a **warning** with wide bands; never silently pass a PR.
8. **Tenant isolation is non-negotiable** *(platform vision)*. Row-level security on OLTP, tenant-scoped OLAP partitions, per-tenant encryption keys for prompt content.

**Non-functional requirements**

| Concern | Target |
|---|---|
| PR check latency (static path) | p95 < 8 s |
| PR check latency (with simulation) | p95 < 45 s |
| Policy evaluation latency | p95 < 50 ms |
| Telemetry ingest throughput | 100k events/s/region sustained |
| Telemetry query (exec dashboard) | p95 < 2 s over 90-day window |
| Availability (control APIs) | 99.9% |
| Prediction calibration | p95 predicted within ±20% of actual after 2 weeks of telemetry |
| Data residency | per-tenant region pinning |

---

## 4. Microservice architecture

Services are independently deployable, own their data, and communicate via the event bus (async) or gRPC (sync internal). Public traffic enters only through the API Gateway.

### 4.1 Service catalog

| Service | Responsibility | Sync deps | Emits | Consumes |
|---|---|---|---|---|
| **api-gateway** | TLS, authn (OIDC/PAT), tenant resolution, rate limit, request routing | — | — | — |
| **analysis-orchestrator** | Owns the pre-deploy pipeline state machine; fans work to engines; aggregates results | model-catalog | `analysis.requested`, `analysis.completed` | `pr.checked` |
| **workflow-parser** | Ingests workflow/agent definitions (LangGraph, CrewAI, OpenAI Assistants, custom DSL, raw prompt templates) → normalized **Workflow IR** (a DAG) | — | `workflow.parsed` | `analysis.requested` |
| **prediction-engine** | Predicts per-node token distribution (input+output) from IR + historical telemetry + heuristics | model-catalog, telemetry-store | `prediction.completed` | `workflow.parsed` |
| **token-sim-engine** | Monte Carlo over traffic scenarios → token-consumption distribution; what-if model swaps for cost comparison | model-catalog, prediction-engine | `simulation.completed` | `prediction.completed` |
| **risk-detectors** | Static + simulated checks: context explosion, recursive loops, fan-out, unbounded retries, prompt bloat | — | `risk.findings` | `prediction.completed`, `simulation.completed` |
| **risk-scorer** | Combines findings + projected token consumption + policy posture → 0–100 deployment risk score + gate decision | policy-engine | `risk.scored` | `risk.findings`, `simulation.completed` |
| **model-catalog** | Source of truth for models, capabilities, context limits, pricing, substitution graph | — | `pricing.updated` | — |
| **policy-engine** | Real-time rule evaluation (token ceilings, allowlists, context caps, loop guards). OPA-style policy bundles per tenant | model-catalog | `policy.violation` | `prediction.completed`, `telemetry.usage` |
| **telemetry-ingest** | OTLP/HTTP ingestion endpoint, schema validation, enrichment, dedupe | — | `telemetry.usage` | — |
| **telemetry-processor** | Streams events → ClickHouse; rollups; drift detection vs predictions | — | `telemetry.rollup`, `drift.detected` | `telemetry.usage` |
| **forecast-service** | Executive token forecasts (time-series + driver decomposition); remediation plan generation | model-catalog, telemetry-store | `forecast.updated` | `telemetry.rollup`, `risk.scored` |
| **github-app** | GitHub App: webhook handling, check-run creation/update, PR comments | analysis-orchestrator | `pr.checked` | `analysis.completed`, `risk.scored` |
| **remediation-service** | Turns findings into concrete, ranked engineering fixes (diffs, config changes, model swaps) | model-catalog | `remediation.ready` | `risk.scored` |
| **tenant-service** | Orgs, projects, members, API keys, billing plan, region pinning | — | `tenant.changed` | — |
| **notification-service** | Slack/email/webhook fan-out for violations, forecasts, gate failures | — | — | `policy.violation`, `forecast.updated`, `risk.scored` |

### 4.2 Data ownership

- **Postgres (OLTP):** tenant-service, model-catalog, analysis-orchestrator, policy-engine, risk-scorer, remediation-service. Strict per-service schemas; no cross-service table reads.
- **ClickHouse (OLAP):** telemetry-processor (writer), forecast-service & dashboards (readers).
- **Object store (S3-compatible):** raw workflow artifacts, large prompt templates, simulation result blobs — referenced by URI from OLTP rows.
- **Redis:** policy bundle cache, model pricing cache, rate-limit counters, idempotency keys.

### 4.3 Sync vs async

- Pre-deploy pipeline is **async by default** (events) so a slow simulation never blocks the gateway. The GitHub check-run is created immediately as `in_progress` and updated on `analysis.completed`.
- The **policy plane is sync** (gRPC) because real-time evaluation must answer within a request budget.

---

## 5. Event-driven ingestion & event model

### 5.1 Bus topology

Kafka (or Redpanda) is the backbone. Topics are tenant-partitioned by key `tenant_id` so per-tenant ordering is preserved and consumers can be scaled per partition. A schema registry (Avro/Protobuf + JSON Schema for external) enforces compatibility.

| Topic | Key | Partitions | Retention | Producers | Consumers |
|---|---|---|---|---|---|
| `ingestion.raw` | `tenant_id` | 64 | 24h | telemetry-ingest | telemetry-processor |
| `telemetry.usage` | `tenant_id` | 64 | 7d | telemetry-ingest | telemetry-processor, policy-engine |
| `telemetry.rollup` | `tenant_id` | 16 | 30d | telemetry-processor | forecast-service |
| `drift.detected` | `tenant_id` | 8 | 30d | telemetry-processor | prediction-engine, notification-service |
| `analysis.requested` | `analysis_id` | 32 | 7d | analysis-orchestrator | workflow-parser |
| `workflow.parsed` | `analysis_id` | 32 | 7d | workflow-parser | prediction-engine |
| `prediction.completed` | `analysis_id` | 32 | 7d | prediction-engine | token-sim-engine, risk-detectors, policy-engine |
| `simulation.completed` | `analysis_id` | 32 | 7d | token-sim-engine | risk-detectors, risk-scorer |
| `risk.findings` | `analysis_id` | 32 | 7d | risk-detectors | risk-scorer |
| `risk.scored` | `analysis_id` | 32 | 30d | risk-scorer | github-app, remediation-service, forecast-service, notification-service |
| `analysis.completed` | `analysis_id` | 32 | 30d | analysis-orchestrator | github-app |
| `policy.violation` | `tenant_id` | 16 | 30d | policy-engine | notification-service, forecast-service |
| `remediation.ready` | `analysis_id` | 16 | 30d | remediation-service | github-app, notification-service |
| `forecast.updated` | `tenant_id` | 8 | 90d | forecast-service | notification-service |
| `pricing.updated` | `global` | 4 | compact | model-catalog | prediction-engine, token-sim-engine, policy-engine |

### 5.2 Envelope

Every event shares an envelope; the `data` payload is type-specific.

```json
{
  "event_id": "evt_01HX...",            // ULID, idempotency key
  "type": "prediction.completed",
  "spec_version": "1.0",
  "occurred_at": "2026-06-05T14:02:11.041Z",
  "tenant_id": "ten_8fa3",
  "trace_id": "4bf92f3577b34da6...",     // W3C traceparent for correlation
  "producer": "prediction-engine@1.4.2",
  "partition_key": "ana_7c19",
  "data": { /* type-specific */ }
}
```

### 5.3 Key payload schemas (JSON Schema, abridged)

**`telemetry.usage`** — one normalized LLM call as observed in production:

```json
{
  "call_id": "call_01HY...",
  "tenant_id": "ten_8fa3",
  "project_id": "prj_22",
  "workflow_id": "wf_payments_agent",
  "deployment_id": "dep_91",
  "node_id": "node_classify",          // maps to Workflow IR node
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "request": {
    "input_tokens": 1843,
    "cached_input_tokens": 1200,
    "output_tokens": 412,
    "context_window_used": 2255,
    "tool_calls": 1,
    "temperature": 0.2
  },
  "latency_ms": 980,
  "cost_usd": 0.00731,                  // computed at ingest via pricing snapshot
  "pricing_version": "pr_2026_06_01",
  "parent_call_id": "call_01HX...",     // for loop/chain reconstruction
  "depth": 3,                           // recursion depth in this trace
  "status": "ok",
  "occurred_at": "2026-06-05T14:01:59Z"
}
```

**`prediction.completed`** — per-node token distribution:

```json
{
  "analysis_id": "ana_7c19",
  "workflow_id": "wf_payments_agent",
  "nodes": [
    {
      "node_id": "node_classify",
      "model": "claude-sonnet-4-6",
      "input_tokens": {"p50": 1700, "p95": 2600, "p99": 3400},
      "output_tokens": {"p50": 380, "p95": 900, "p99": 1500},
      "expected_calls_per_request": {"p50": 1, "p95": 1, "p99": 2},
      "confidence": 0.81,
      "basis": "telemetry+heuristic"     // telemetry | heuristic | tokenizer_static
    }
  ],
  "request_tokens": {"p50": 2080, "p95": 3500, "p99": 4900}
}
```

**`risk.findings`** — structured detector output:

```json
{
  "analysis_id": "ana_7c19",
  "findings": [
    {
      "finding_id": "f_ctx_01",
      "category": "context_explosion",
      "severity": "high",
      "node_id": "node_summarize",
      "evidence": {
        "growth_pattern": "linear_accumulation",
        "per_iteration_token_delta": 1400,
        "unbounded": true,
        "projected_tokens_at_iter_20": 31200,
        "model_context_limit": 200000
      },
      "message": "Conversation history is appended without truncation; context grows ~1.4K tokens/iteration."
    },
    {
      "finding_id": "f_loop_02",
      "category": "recursive_loop",
      "severity": "critical",
      "evidence": {
        "cycle": ["node_plan", "node_act", "node_reflect", "node_plan"],
        "termination_guard": "none_detected",
        "max_observed_depth": 41
      },
      "message": "Cycle plan→act→reflect has no bounded termination condition."
    }
  ]
}
```

**`risk.scored`**:

```json
{
  "analysis_id": "ana_7c19",
  "score": 82,
  "band": "high",
  "gate_decision": "block",            // pass | warn | block
  "drivers": [
    {"category": "recursive_loop", "contribution": 38},
    {"category": "context_explosion", "contribution": 24},
    {"category": "token_ceiling_exceeded", "contribution": 20}
  ],
  "projected_monthly_tokens": {"p50": 4100000000, "p95": 18800000000},
  "policy_refs": ["pol_token_ceiling_prod", "pol_model_allowlist"]
}
```

### 5.4 Delivery semantics

- **At-least-once** delivery; consumers are **idempotent** keyed on `event_id` (dedupe table / Redis set with TTL).
- **Exactly-once-ish** for telemetry cost aggregation via Kafka transactions on the processor → ClickHouse insert path, with `call_id` as the natural dedupe key.
- **Ordering** guaranteed per partition key only. Pipeline stages tolerate out-of-order arrival by joining on `analysis_id` and waiting for required predecessors (orchestrator tracks a completion set).
- **DLQ** per consumer group; poison messages parked after N retries with exponential backoff.

---

## 6. Conceptual data model

Core entities and relationships (crow's-foot in prose):

- **Tenant** 1—N **Project** 1—N **Workflow**. A Workflow is a logical agent/pipeline.
- **Workflow** 1—N **WorkflowVersion** (immutable; each parse produces a version with a content hash). A WorkflowVersion *has* a **Workflow IR** (the normalized DAG) stored as nodes + edges.
- **WorkflowIRNode** N—1 **Model** (intended model). **WorkflowIREdge** carries control-flow type (sequence, conditional, loop).
- **Analysis** belongs to a WorkflowVersion and a trigger (PR, manual, scheduled). An Analysis *produces* one **Prediction**, zero-or-more **Simulations**, N **Findings**, one **RiskScore**, and zero-or-one **RemediationPlan**.
- **Model** belongs to a **Provider**, has many **PricePoints** (time-versioned), and **SubstitutionEdges** to cheaper/comparable models with a capability-compatibility score.
- **Policy** belongs to a Tenant, scoped to project/workflow/env, with a typed rule body. **PolicyViolation** links a Policy to an Analysis or a live telemetry window.
- **Deployment** links a WorkflowVersion to an environment; **UsageEvent** (telemetry, in OLAP) references deployment + node and is the recalibration substrate.
- **Forecast** belongs to a Tenant/Project, derived from rollups + scored analyses.

OLTP holds everything except high-volume **UsageEvent** rows, which live in ClickHouse and are referenced by id only.

---

## 7. Database schema

### 7.1 OLTP — PostgreSQL DDL

Multi-tenancy via a `tenant_id` column on every table + Postgres **Row-Level Security**. Connection pool sets `SET app.tenant_id` per request; policies filter on it.

```sql
-- ============ Tenancy & identity ============
CREATE TABLE tenant (
  tenant_id        TEXT PRIMARY KEY,            -- ULID, e.g. ten_8fa3
  name             TEXT NOT NULL,
  plan             TEXT NOT NULL DEFAULT 'trial',
  region           TEXT NOT NULL DEFAULT 'us-east-1',
  data_residency   TEXT NOT NULL DEFAULT 'us',
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  status           TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE project (
  project_id   TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  name         TEXT NOT NULL,
  repo_url     TEXT,
  default_env  TEXT NOT NULL DEFAULT 'production',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, name)
);

CREATE TABLE member (
  member_id  TEXT PRIMARY KEY,
  tenant_id  TEXT NOT NULL REFERENCES tenant(tenant_id),
  email      TEXT NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('owner','admin','engineer','viewer')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, email)
);

CREATE TABLE api_key (
  api_key_id  TEXT PRIMARY KEY,
  tenant_id   TEXT NOT NULL REFERENCES tenant(tenant_id),
  hash        TEXT NOT NULL,                    -- argon2 of the token
  scopes      TEXT[] NOT NULL DEFAULT '{}',
  last_used_at TIMESTAMPTZ,
  revoked_at  TIMESTAMPTZ,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============ Provider / model catalog ============
CREATE TABLE provider (
  provider_id TEXT PRIMARY KEY,                 -- openai, anthropic, google, oss
  name        TEXT NOT NULL,
  kind        TEXT NOT NULL CHECK (kind IN ('hosted','self_hosted'))
);

CREATE TABLE model (
  model_id        TEXT PRIMARY KEY,             -- e.g. claude-sonnet-4-6
  provider_id     TEXT NOT NULL REFERENCES provider(provider_id),
  family          TEXT NOT NULL,
  context_limit   INTEGER NOT NULL,
  max_output      INTEGER NOT NULL,
  modality        TEXT[] NOT NULL DEFAULT '{text}',
  supports_tools  BOOLEAN NOT NULL DEFAULT true,
  tokenizer       TEXT NOT NULL,                -- cl100k, o200k, claude, sentencepiece...
  quality_tier    SMALLINT NOT NULL,            -- 1..5 capability tier
  status          TEXT NOT NULL DEFAULT 'ga',   -- ga | preview | deprecated
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE price_point (
  price_point_id   TEXT PRIMARY KEY,
  model_id         TEXT NOT NULL REFERENCES model(model_id),
  pricing_version  TEXT NOT NULL,               -- pr_2026_06_01
  input_per_mtok   NUMERIC(12,4) NOT NULL,      -- USD per 1M input tokens
  cached_input_per_mtok NUMERIC(12,4),
  output_per_mtok  NUMERIC(12,4) NOT NULL,
  effective_from   TIMESTAMPTZ NOT NULL,
  effective_to     TIMESTAMPTZ,
  UNIQUE (model_id, pricing_version)
);

CREATE TABLE substitution_edge (
  from_model_id    TEXT NOT NULL REFERENCES model(model_id),
  to_model_id      TEXT NOT NULL REFERENCES model(model_id),
  capability_score NUMERIC(4,3) NOT NULL,       -- 0..1, how safe the swap is
  avg_cost_ratio   NUMERIC(6,3) NOT NULL,       -- to/from expected cost
  notes            TEXT,
  PRIMARY KEY (from_model_id, to_model_id)
);

-- ============ Workflows & IR ============
CREATE TABLE workflow (
  workflow_id  TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  project_id   TEXT NOT NULL REFERENCES project(project_id),
  name         TEXT NOT NULL,
  source_kind  TEXT NOT NULL,                   -- langgraph|crewai|openai_assistants|dsl|prompt
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (project_id, name)
);

CREATE TABLE workflow_version (
  workflow_version_id TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  workflow_id  TEXT NOT NULL REFERENCES workflow(workflow_id),
  content_hash TEXT NOT NULL,                   -- sha256 of source artifact
  artifact_uri TEXT NOT NULL,                   -- s3://.../source
  ir_uri       TEXT NOT NULL,                   -- s3://.../ir.json (full DAG)
  git_sha      TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (workflow_id, content_hash)
);

-- IR nodes/edges denormalized into OLTP for query; full IR also in S3.
CREATE TABLE ir_node (
  workflow_version_id TEXT NOT NULL REFERENCES workflow_version(workflow_version_id),
  node_id      TEXT NOT NULL,
  kind         TEXT NOT NULL,                   -- llm_call|tool|router|map|reduce|human
  intended_model_id TEXT REFERENCES model(model_id),
  prompt_template_uri TEXT,
  static_input_tokens INTEGER,                  -- tokenizer count of static prompt
  appends_history BOOLEAN NOT NULL DEFAULT false,
  has_termination_guard BOOLEAN,
  PRIMARY KEY (workflow_version_id, node_id)
);

CREATE TABLE ir_edge (
  workflow_version_id TEXT NOT NULL REFERENCES workflow_version(workflow_version_id),
  from_node    TEXT NOT NULL,
  to_node      TEXT NOT NULL,
  edge_type    TEXT NOT NULL CHECK (edge_type IN ('sequence','conditional','loop','fanout')),
  condition    TEXT,
  PRIMARY KEY (workflow_version_id, from_node, to_node)
);

-- ============ Analysis pipeline ============
CREATE TABLE analysis (
  analysis_id  TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  workflow_version_id TEXT NOT NULL REFERENCES workflow_version(workflow_version_id),
  trigger      TEXT NOT NULL CHECK (trigger IN ('pr','manual','scheduled','api')),
  git_pr_number INTEGER,
  status       TEXT NOT NULL DEFAULT 'queued',  -- queued|running|completed|failed
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ
);

CREATE TABLE prediction (
  prediction_id TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  analysis_id  TEXT NOT NULL REFERENCES analysis(analysis_id),
  request_tokens_p50 NUMERIC(14,2),
  request_tokens_p95 NUMERIC(14,2),
  request_tokens_p99 NUMERIC(14,2),
  basis        TEXT NOT NULL,                   -- telemetry|heuristic|tokenizer_static
  confidence   NUMERIC(4,3) NOT NULL,
  detail_uri   TEXT NOT NULL,                   -- per-node distributions blob
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE simulation (
  simulation_id TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  analysis_id  TEXT NOT NULL REFERENCES analysis(analysis_id),
  scenario_name TEXT NOT NULL,                  -- e.g. "peak_2x", "steady_state"
  rps          NUMERIC(10,2) NOT NULL,
  horizon_days INTEGER NOT NULL,
  model_overrides JSONB,                        -- what-if substitutions
  monthly_tokens_p50 NUMERIC(18,2),
  monthly_tokens_p95 NUMERIC(18,2),
  monthly_tokens_p99 NUMERIC(18,2),
  detail_uri   TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE finding (
  finding_id   TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  analysis_id  TEXT NOT NULL REFERENCES analysis(analysis_id),
  category     TEXT NOT NULL,                   -- context_explosion|recursive_loop|fanout|prompt_bloat|retry_storm|model_mismatch
  severity     TEXT NOT NULL CHECK (severity IN ('low','medium','high','critical')),
  node_id      TEXT,
  evidence     JSONB NOT NULL,
  message      TEXT NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE risk_score (
  risk_score_id TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  analysis_id  TEXT NOT NULL UNIQUE REFERENCES analysis(analysis_id),
  score        SMALLINT NOT NULL CHECK (score BETWEEN 0 AND 100),
  band         TEXT NOT NULL CHECK (band IN ('low','medium','high','critical')),
  gate_decision TEXT NOT NULL CHECK (gate_decision IN ('pass','warn','block')),
  drivers      JSONB NOT NULL,
  projected_monthly_tokens_p50 NUMERIC(18,2),
  projected_monthly_tokens_p95 NUMERIC(18,2),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE remediation_plan (
  remediation_plan_id TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  analysis_id  TEXT NOT NULL REFERENCES analysis(analysis_id),
  items        JSONB NOT NULL,                  -- ranked fixes w/ est. token savings
  est_monthly_tokens_saved NUMERIC(18,2),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============ Policy ============
CREATE TABLE policy (
  policy_id    TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  scope_project TEXT REFERENCES project(project_id),
  scope_env    TEXT,                            -- null = all envs
  name         TEXT NOT NULL,
  type         TEXT NOT NULL,                   -- token_ceiling|model_allowlist|context_cap|loop_guard|gate_threshold
  rule         JSONB NOT NULL,                  -- typed body, see §9.6
  enforcement  TEXT NOT NULL CHECK (enforcement IN ('warn','block')),
  enabled      BOOLEAN NOT NULL DEFAULT true,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE policy_violation (
  violation_id TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  policy_id    TEXT NOT NULL REFERENCES policy(policy_id),
  analysis_id  TEXT REFERENCES analysis(analysis_id),
  source       TEXT NOT NULL CHECK (source IN ('predeploy','runtime')),
  observed     JSONB NOT NULL,
  occurred_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============ Deployments & forecasts ============
CREATE TABLE deployment (
  deployment_id TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  workflow_version_id TEXT NOT NULL REFERENCES workflow_version(workflow_version_id),
  environment  TEXT NOT NULL,
  active       BOOLEAN NOT NULL DEFAULT true,
  deployed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE forecast (
  forecast_id  TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL REFERENCES tenant(tenant_id),
  scope_project TEXT REFERENCES project(project_id),
  horizon_days INTEGER NOT NULL,
  method       TEXT NOT NULL,                   -- prophet|ewma|driver_decomp
  points_uri   TEXT NOT NULL,                   -- time-series blob
  total_monthly_tokens_p50 NUMERIC(18,2),
  total_monthly_tokens_p95 NUMERIC(18,2),
  drivers      JSONB,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============ RLS example ============
ALTER TABLE workflow ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON workflow
  USING (tenant_id = current_setting('app.tenant_id')::text);
-- (repeat per tenant-scoped table)

-- Hot-path indexes
CREATE INDEX idx_analysis_tenant_status ON analysis(tenant_id, status);
CREATE INDEX idx_finding_analysis ON finding(analysis_id);
CREATE INDEX idx_price_point_model_eff ON price_point(model_id, effective_from DESC);
CREATE INDEX idx_policy_tenant_enabled ON policy(tenant_id, enabled);
CREATE INDEX idx_irnode_wv ON ir_node(workflow_version_id);
```

### 7.2 OLAP — ClickHouse telemetry warehouse

```sql
CREATE TABLE usage_event (
  tenant_id        LowCardinality(String),
  project_id       LowCardinality(String),
  workflow_id      LowCardinality(String),
  deployment_id    LowCardinality(String),
  node_id          LowCardinality(String),
  call_id          String,
  parent_call_id   String,
  trace_id         String,
  depth            UInt16,
  provider         LowCardinality(String),
  model            LowCardinality(String),
  input_tokens     UInt32,
  cached_input_tokens UInt32,
  output_tokens    UInt32,
  context_window_used UInt32,
  tool_calls       UInt16,
  latency_ms       UInt32,
  cost_usd         Decimal(18,8),
  pricing_version  LowCardinality(String),
  status           LowCardinality(String),
  occurred_at      DateTime64(3)
)
ENGINE = MergeTree
PARTITION BY (tenant_id, toYYYYMMDD(occurred_at))
ORDER BY (tenant_id, workflow_id, node_id, occurred_at)
TTL toDateTime(occurred_at) + INTERVAL 400 DAY;

-- Hourly rollup powering dashboards & forecasts
CREATE MATERIALIZED VIEW usage_hourly_mv
ENGINE = SummingMergeTree
PARTITION BY (tenant_id, toYYYYMM(hour))
ORDER BY (tenant_id, workflow_id, node_id, model, hour)
AS SELECT
  tenant_id, workflow_id, node_id, model,
  toStartOfHour(occurred_at) AS hour,
  count() AS calls,
  sum(input_tokens) AS input_tokens,
  sum(output_tokens) AS output_tokens,
  sum(cost_usd) AS cost_usd,
  max(depth) AS max_depth,
  max(context_window_used) AS max_context
FROM usage_event
GROUP BY tenant_id, workflow_id, node_id, model, hour;
```

The hourly MV is the recalibration source for the Prediction Engine (per-node empirical distributions) and the input to the Forecast Service. `max_depth` and `max_context` here also drive **runtime** loop/context drift detection.

---

## 8. API contracts

Public REST API, versioned at `/v1`. JSON over HTTPS. Auth via `Authorization: Bearer <api_key>` (PAT) or GitHub App installation token. All list endpoints are cursor-paginated. All write endpoints accept an `Idempotency-Key` header.

### 8.1 OpenAPI 3.1 (core surface, abridged)

```yaml
openapi: 3.1.0
info:
  title: Tollgate API
  version: "1.0.0"
servers:
  - url: https://api.tollgate.com/v1
security:
  - bearerAuth: []
paths:
  /workflows:
    post:
      summary: Register or update a workflow source (triggers a parse)
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/WorkflowCreate' }
      responses:
        '201':
          description: Workflow version created
          content:
            application/json:
              schema: { $ref: '#/components/schemas/WorkflowVersion' }

  /analyses:
    post:
      summary: Start an analysis for a workflow version
      parameters:
        - in: header
          name: Idempotency-Key
          schema: { type: string }
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [workflow_version_id, trigger]
              properties:
                workflow_version_id: { type: string }
                trigger: { type: string, enum: [pr, manual, scheduled, api] }
                git_pr_number: { type: integer }
                scenarios:
                  type: array
                  items: { $ref: '#/components/schemas/TrafficScenario' }
                run_simulation: { type: boolean, default: true }
      responses:
        '202':
          description: Analysis accepted (async)
          content:
            application/json:
              schema: { $ref: '#/components/schemas/Analysis' }

    get:
      summary: List analyses
      parameters:
        - in: query
          name: workflow_id
          schema: { type: string }
        - in: query
          name: cursor
          schema: { type: string }
      responses:
        '200':
          description: Page of analyses
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items: { $ref: '#/components/schemas/Analysis' }
                  next_cursor: { type: string, nullable: true }

  /analyses/{analysis_id}:
    get:
      summary: Get analysis with full result bundle
      parameters:
        - in: path
          name: analysis_id
          required: true
          schema: { type: string }
      responses:
        '200':
          description: Analysis result
          content:
            application/json:
              schema: { $ref: '#/components/schemas/AnalysisResult' }

  /analyses/{analysis_id}/simulate:
    post:
      summary: Run an additional what-if simulation (e.g. model swap)
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                scenarios:
                  type: array
                  items: { $ref: '#/components/schemas/TrafficScenario' }
                model_overrides:
                  type: object
                  additionalProperties: { type: string }   # node_id -> model_id
      responses:
        '200':
          content:
            application/json:
              schema: { $ref: '#/components/schemas/SimulationResult' }

  /policies:
    get:
      summary: List policies
      responses:
        '200':
          content:
            application/json:
              schema:
                type: array
                items: { $ref: '#/components/schemas/Policy' }
    post:
      summary: Create a policy
      requestBody:
        content:
          application/json:
            schema: { $ref: '#/components/schemas/Policy' }
      responses:
        '201':
          content:
            application/json:
              schema: { $ref: '#/components/schemas/Policy' }

  /policies/evaluate:
    post:
      summary: Real-time policy evaluation of a proposed request shape (sync, <50ms)
      requestBody:
        content:
          application/json:
            schema: { $ref: '#/components/schemas/EvaluateRequest' }
      responses:
        '200':
          content:
            application/json:
              schema: { $ref: '#/components/schemas/EvaluateResponse' }

  /telemetry/usage:
    post:
      summary: Ingest a batch of usage events (OTLP-JSON or native)
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                events:
                  type: array
                  items: { $ref: '#/components/schemas/UsageEvent' }
      responses:
        '202': { description: Accepted }

  /forecasts:
    get:
      summary: Get executive token forecast
      parameters:
        - in: query
          name: project_id
          schema: { type: string }
        - in: query
          name: horizon_days
          schema: { type: integer, default: 30 }
      responses:
        '200':
          content:
            application/json:
              schema: { $ref: '#/components/schemas/Forecast' }

  /models:
    get:
      summary: List supported models with current pricing
      responses:
        '200':
          content:
            application/json:
              schema:
                type: array
                items: { $ref: '#/components/schemas/Model' }

components:
  securitySchemes:
    bearerAuth: { type: http, scheme: bearer }
  schemas:
    WorkflowCreate:
      type: object
      required: [project_id, name, source_kind, source]
      properties:
        project_id: { type: string }
        name: { type: string }
        source_kind:
          type: string
          enum: [langgraph, crewai, openai_assistants, dsl, prompt]
        source: { type: string, description: "Inline source or s3:// URI" }
        git_sha: { type: string }
    WorkflowVersion:
      type: object
      properties:
        workflow_version_id: { type: string }
        workflow_id: { type: string }
        content_hash: { type: string }
        node_count: { type: integer }
        created_at: { type: string, format: date-time }
    TrafficScenario:
      type: object
      required: [name, rps, horizon_days]
      properties:
        name: { type: string }
        rps: { type: number }
        horizon_days: { type: integer }
        diurnal_peak_multiplier: { type: number, default: 1.0 }
    Analysis:
      type: object
      properties:
        analysis_id: { type: string }
        status: { type: string, enum: [queued, running, completed, failed] }
        created_at: { type: string, format: date-time }
    AnalysisResult:
      type: object
      properties:
        analysis: { $ref: '#/components/schemas/Analysis' }
        prediction: { $ref: '#/components/schemas/Prediction' }
        simulations:
          type: array
          items: { $ref: '#/components/schemas/SimulationResult' }
        findings:
          type: array
          items: { $ref: '#/components/schemas/Finding' }
        risk_score: { $ref: '#/components/schemas/RiskScore' }
        remediation_plan: { $ref: '#/components/schemas/RemediationPlan' }
    Prediction:
      type: object
      properties:
        request_tokens:
          $ref: '#/components/schemas/Distribution'
        confidence: { type: number }
        basis: { type: string, enum: [telemetry, heuristic, tokenizer_static] }
        nodes:
          type: array
          items:
            type: object
            properties:
              node_id: { type: string }
              model: { type: string }
              input_tokens: { $ref: '#/components/schemas/Distribution' }
              output_tokens: { $ref: '#/components/schemas/Distribution' }
    Distribution:
      type: object
      properties:
        p50: { type: number }
        p95: { type: number }
        p99: { type: number }
    SimulationResult:
      type: object
      properties:
        simulation_id: { type: string }
        scenario_name: { type: string }
        monthly_tokens: { $ref: '#/components/schemas/Distribution' }
    Finding:
      type: object
      properties:
        finding_id: { type: string }
        category:
          type: string
          enum: [context_explosion, recursive_loop, fanout, prompt_bloat, retry_storm, model_mismatch]
        severity: { type: string, enum: [low, medium, high, critical] }
        node_id: { type: string }
        evidence: { type: object, additionalProperties: true }
        message: { type: string }
    RiskScore:
      type: object
      properties:
        score: { type: integer, minimum: 0, maximum: 100 }
        band: { type: string, enum: [low, medium, high, critical] }
        gate_decision: { type: string, enum: [pass, warn, block] }
        drivers:
          type: array
          items:
            type: object
            properties:
              category: { type: string }
              contribution: { type: number }
        projected_monthly_tokens: { $ref: '#/components/schemas/Distribution' }
    RemediationPlan:
      type: object
      properties:
        est_monthly_tokens_saved: { type: number }
        items:
          type: array
          items:
            type: object
            properties:
              title: { type: string }
              category: { type: string }
              est_monthly_tokens_saved: { type: number }
              effort: { type: string, enum: [low, medium, high] }
              suggested_diff: { type: string }
    Policy:
      type: object
      required: [name, type, rule, enforcement]
      properties:
        policy_id: { type: string, readOnly: true }
        name: { type: string }
        scope_project: { type: string, nullable: true }
        scope_env: { type: string, nullable: true }
        type:
          type: string
          enum: [token_ceiling, model_allowlist, context_cap, loop_guard, gate_threshold]
        rule: { type: object, additionalProperties: true }
        enforcement: { type: string, enum: [warn, block] }
        enabled: { type: boolean }
    EvaluateRequest:
      type: object
      properties:
        project_id: { type: string }
        environment: { type: string }
        request_shape:
          type: object
          properties:
            model: { type: string }
            estimated_input_tokens: { type: integer }
            estimated_output_tokens: { type: integer }
            depth: { type: integer }
            context_window_used: { type: integer }
    EvaluateResponse:
      type: object
      properties:
        decision: { type: string, enum: [allow, warn, deny] }
        violations:
          type: array
          items:
            type: object
            properties:
              policy_id: { type: string }
              type: { type: string }
              message: { type: string }
    UsageEvent:
      type: object
      required: [call_id, model, provider]
      properties:
        call_id: { type: string }
        parent_call_id: { type: string }
        workflow_id: { type: string }
        node_id: { type: string }
        deployment_id: { type: string }
        provider: { type: string }
        model: { type: string }
        input_tokens: { type: integer }
        cached_input_tokens: { type: integer }
        output_tokens: { type: integer }
        context_window_used: { type: integer }
        depth: { type: integer }
        latency_ms: { type: integer }
        status: { type: string }
        occurred_at: { type: string, format: date-time }
    Model:
      type: object
      properties:
        model_id: { type: string }
        provider: { type: string }
        context_limit: { type: integer }
        quality_tier: { type: integer }
        input_per_mtok: { type: number }
        output_per_mtok: { type: number }
    Forecast:
      type: object
      properties:
        horizon_days: { type: integer }
        total_monthly_tokens: { $ref: '#/components/schemas/Distribution' }
        drivers:
          type: array
          items:
            type: object
            properties:
              workflow_id: { type: string }
              share: { type: number }
              trend: { type: string, enum: [rising, flat, falling] }
```

### 8.2 Internal service contracts (gRPC, sketched)

```protobuf
// prediction-engine
service Prediction {
  rpc Predict(PredictRequest) returns (PredictResponse);
}
message PredictRequest {
  string tenant_id = 1;
  string workflow_version_id = 2;
  bytes  ir = 3;                       // serialized Workflow IR
}
message PredictResponse {
  repeated NodePrediction nodes = 1;
  Distribution request_tokens = 2;
  double confidence = 3;
  string basis = 4;
}

// policy-engine (sync, hot path)
service Policy {
  rpc Evaluate(EvaluateRequest) returns (EvaluateResponse);   // p95 < 50ms
}
```

### 8.3 Webhooks (outbound to customer)

`POST {tenant_webhook_url}` with the same event envelope as §5.2, signed with `X-AITC-Signature: sha256=...` (HMAC over body using the tenant webhook secret). Delivered for `risk.scored`, `policy.violation`, `forecast.updated`, `drift.detected`.

---

## 9. Core engines

This is where prevention is actually delivered. Each engine maps to one or more of the ten required capabilities.

### 9.1 Workflow Parser → Workflow IR (capability 1)

The parser normalizes heterogeneous agent definitions into one **Workflow IR**: a directed graph where nodes are LLM calls / tools / routers / map-reduce / human steps, and edges carry control-flow type (sequence, conditional, loop, fanout).

Adapters (all implemented via Python `ast`, never executing the target):

- **LangGraph:** graph-builder introspection (`StateGraph`/`add_node`/`add_edge`/`add_conditional_edges`/`set_entry_point`) to recover nodes and edges; back-edges become loop edges; models recovered from source.
- **CrewAI:** `Agent`/`Task`/`Crew` recovered into a task graph; sequential process → sequence edges; **hierarchical process or any `allow_delegation` agent → an unguarded delegation loop edge** (CrewAI has no built-in delegation cap, so it's flagged), `memory=True` → appends-history. Handles both the plain and `@CrewBase`/`@agent`/`@task` decorator styles.
- **AutoGPT:** block-graph JSON exports → nodes/edges.
- **Hand-rolled imperative agents (framework-agnostic):** a `while`/`for` loop around a recognized LLM SDK call (BabyAGI/AutoGPT-class). Recognized SDK surfaces include OpenAI and OpenAI-compatible vendors (Azure, Groq, Together, DeepSeek, Fireworks, OpenRouter, xAI, Perplexity, vLLM, LM Studio, Ollama `/v1`), Anthropic, Google Gemini/Vertex, Mistral, Amazon Bedrock (`converse`/`invoke_model`), Cohere, Replicate, Ollama, Hugging Face, and LiteLLM.
- **Native DSL / raw prompt templates:** YAML/JSON workflow schema or a single templated prompt → IR with variable slots.

A repo's discovery is widened to surface files of frameworks Tollgate recognizes but does not fully graph (LangChain `AgentExecutor`, AutoGen, LlamaIndex, smolagents, …); those route to the strict linter (§9.8) rather than being dropped.

For each `llm_call` node the parser computes the **static prompt token count** with the model's tokenizer, identifies whether the node **appends history** (the seed of context explosion), and records whether any **termination guard** is present on loop edges (counter, max-depth, confidence threshold, stop token). A parse that yields no LLM node and no edges is treated as **honest failure** and dropped — never scored as a confident PASS.

Output (OSS): an in-memory Workflow IR per file. *(Platform vision: IR persisted to S3 + denormalized into `ir_node`/`ir_edge`.)* Deterministic and fast — it alone powers the sub-second static path.

### 9.2 Prediction Engine (capability 2)

Predicts a **distribution** of input and output tokens per node, then composes them along the graph.

Three bases, chosen per node by confidence:

1. **Telemetry-backed (best):** empirical p50/p95/p99 from the hourly MV for this node (or a similar node by prompt-template hash) on this or a sibling workflow. Used when ≥ N historical calls exist.
2. **Heuristic:** regression on features — static prompt tokens, retrieved-context size, tool-output size priors, model family output verbosity priors.
3. **Tokenizer-static (floor):** exact static token count for input; output modeled from `max_output` and historical output/input ratios for the family. Lowest confidence, widest bands.

Graph composition: expected calls per node = product of branch probabilities and loop iteration estimates along incoming paths. Loop iteration counts come from telemetry depth distributions when available, else from declared guards, else flagged as **unbounded** (which forces a finding). Per-request **token** distribution = Monte Carlo (§9.3) over node distributions; dollar cost is a secondary view obtained by applying Model Catalog pricing to the same token paths.

**Recalibration loop:** `drift.detected` events (predicted p95 vs observed p95 diverging beyond threshold) retrain the per-node priors nightly; the `basis` and `confidence` reported on each prediction make calibration auditable.

### 9.3 Token Simulation Engine (capability 3)

Given node distributions + a **TrafficScenario** (RPS, horizon, diurnal peak multiplier), runs **Monte Carlo** (default 10k trials):

- Each trial samples node token counts and branch/loop outcomes from their distributions, walks the IR, sums **tokens** consumed.
- Aggregates per-request tokens into p50/p95/p99, then scales by `rps × horizon × diurnal profile` to a **token-consumption** distribution over the horizon. Dollar cost is a derived secondary view (tokens × pricing).
- Supports **what-if model overrides** (`node_id → model_id`): the token volume is unchanged; re-pricing the same sampled token paths under substitute models quantifies the **cost** difference — an apples-to-apples swap comparison.

Default traffic base: a single **`steady_state` scenario of 10,000 requests/week** over a 30-day horizon (no peak/viral multipliers by default). Override per run with `--traffic-per-week` / `--traffic-per-day` (and `--horizon-days`) on the CLI, the field-study runner, and `scripts/scan-github-repo.sh`; additional scenarios can be declared in `.tollgate.yml`. Results cached by `(workflow_version, scenario, pricing_version, overrides_hash)`.

### 9.4 Risk Detectors (capabilities 4 & 5)

Static + simulated structural analysis. Each detector emits `Finding`s with machine-readable `evidence`.

**Context explosion detector (cap 4):**
- Walks loop/recursive paths where a node `appends_history = true` without a truncation/summarization step.
- Models per-iteration token delta and projects context size at iteration *k*; if projection crosses a fraction (e.g. 60%) of the model `context_limit` within a plausible iteration count, emits `context_explosion`.
- Also flags single-shot **prompt bloat**: static prompt already > threshold, or retrieved-context injection with no cap.
- Severity scales with projected overflow and token impact.

**Recursive loop detector (cap 5):**
- Runs cycle detection (Tarjan SCC) on the IR. For each cycle, checks every edge for a **termination guard** (counter decrement, max-depth, confidence/stop condition).
- A cycle with no provable bound → `recursive_loop`, `critical`. A cycle with a guard but a high/uncapped bound → `high`.
- Cross-checks telemetry: if observed `max_depth` for the workflow already exceeds the declared/implied bound, escalates severity (this is the "it already ran 41 deep in prod" signal).
- Also detects **retry storms**: tool/error edges that re-enter an LLM node without backoff or attempt cap.

**Fan-out detector:** map/parallel nodes whose fan-out degree is driven by unbounded input size → multiplicative cost risk.

**Model mismatch detector (feeds cap 6):** a high-tier (expensive) model on a node whose task profile (short, classification-like, deterministic) is well-served by a cheaper tier.

### 9.5 Model Substitution Recommender (capability 6)

Uses the `substitution_edge` graph in Model Catalog. For each `llm_call` node:

- Candidate substitutes = models reachable via edges with `capability_score ≥ threshold` for the node's task class, respecting `supports_tools`, `context_limit`, modality, and any `model_allowlist` policy.
- Ranks candidates by **expected savings** (re-priced via §9.3 what-if) × `capability_score`, discounting by a configurable risk aversion.
- Only recommends a swap when projected savings exceed a floor **and** capability score clears the node's minimum quality tier. Output feeds the remediation plan with a concrete before/after model and cost delta (token volume unchanged — this is a pure cost lever).

Cross-provider aware: an OpenAI node may be recommended an Anthropic or open-source substitute; OSS (self-hosted) candidates carry an amortized infra cost model rather than per-token pricing.

### 9.6 Risk Scorer & gate decision (capability 7)

Produces a 0–100 score and a `pass | warn | block` gate. Score is a weighted aggregation:

```
score = clamp( Σ_finding  severity_weight(finding) × confidence   (saturating)
             + policy_pressure(violations) , 0, 100)
```

- `severity_weight`: critical=40, high=24, medium=10, low=3 (saturating, not purely additive — diminishing returns past one critical).
- `policy_pressure`: hard contribution per `block`-enforcement violation. **Token ceilings** flow through here: a `token_ceiling` policy whose projected-p95 token volume exceeds its limit is a violation, so token-waste pressure enters the score via `policy_pressure` rather than a separate cost term.

Bands: 0–24 low, 25–49 medium, 50–74 high, 75–100 critical. Gate mapping is **policy-driven** (`gate_threshold` policy), default: `block` if any `block`-enforcement policy is violated or score ≥ 75; `warn` if score ≥ 50; else `pass`. `drivers` decomposition is always returned so engineers see *why*.

### 9.6.1 Policy rule bodies (typed `rule` JSONB)

```jsonc
// token_ceiling
{ "max_monthly_tokens": 2000000000, "applies_to": "project", "metric": "projected_p95" }
// token_ceiling (per-request variant)
{ "max_tokens_per_request": 50000, "scope": "request", "metric": "projected_p95" }
// model_allowlist
{ "allow": ["claude-sonnet-4-6","gpt-5-mini","llama-3.3-70b"], "deny_tier_above": 4 }
// context_cap
{ "max_context_tokens": 60000, "scope": "node" }
// loop_guard
{ "require_termination_guard": true, "max_depth": 10 }
// gate_threshold
{ "block_at_score": 75, "warn_at_score": 50, "block_on_block_violation": true }
```

The Policy Engine evaluates these both **pre-deploy** (against a prediction) and at **runtime** (against a live `request_shape` or telemetry window), giving one rule language for both planes.

### 9.7 Forecast Service & remediation (capability 10)

**Executive forecast:** from hourly rollups, fits a time series per driver (workflow/project) — EWMA baseline plus driver decomposition (which workflows/models drive growth, and trend direction). Blends in the **projected token consumption** of analyses that are scored but not yet reflected in telemetry, so a forecast accounts for what's about to ship. Output: total p50/p95 **tokens** over horizon + ranked drivers (dollar cost derived via pricing as a secondary view), surfaced to the exec dashboard and `/forecasts`.

**Engineering remediation plan:** for each high/critical finding, the Remediation Service emits a ranked, concrete fix: add a `max_depth=N` guard on the identified cycle, insert a history-summarization step before the flagged node, set an explicit `max_iter`/`max_iterations`/`recursion_limit`, cap output tokens (`max_tokens=…`), bound an `asyncio.gather` fan-out with a Semaphore, swap model X→Y (cost lever; token volume unchanged), cap retrieved context, add retry backoff. Each item carries estimated monthly **token savings** (or "uncapped" when no bound exists), effort (low/medium/high), and where possible a `suggested_change`. Identical fixes are de-duplicated. This is the bridge from "your PR is risky" to "here's exactly what to change."

### 9.8 Strict agentic linter (the last gate before prod)

Where the graph detectors reason about a *recovered graph*, the strict linter reasons directly about *source* with the AST. It is the catch-all that makes coverage exhaustive: it reaches the many frameworks Tollgate recognizes but can't (yet) graph, and it flags config-absence risks that don't need a graph. Every check is agentic-gated (a known framework import or a recognized LLM SDK call); on non-agentic code it is silent. Findings are **structural only — no token/cost number is invented** for files with no recoverable graph.

Checks (deterministic):

- **Unbounded loop** — a `while True:` driving an LLM call with no `break`/`return` and no cap → critical (block).
- **Missing iteration/recursion cap** — a known agent constructor/runner invoked without its bound: LangChain `AgentExecutor`/`initialize_agent` (`max_iterations`), AutoGen `GroupChat`/`RoundRobinGroupChat`/`initiate_chat` (`max_round`/`max_turns`), LlamaIndex `ReActAgent` (`max_iterations`), smolagents `CodeAgent` (`max_steps`), CrewAI `Agent` (`max_iter`), LangGraph run config (`recursion_limit`) → warn (these have framework defaults, so it's "set an explicit bound," not a hard block).
- **Uncapped output tokens** — a recognized LLM call **or** a LangChain/LlamaIndex model wrapper (`ChatOpenAI(...)`, `OpenAILike(...)`, resolved by import origin to avoid the raw-SDK-client name collision) built with no `max_tokens`/`max_output_tokens`/… → warn.
- **Unbounded fan-out** — `asyncio.gather` over an input-driven comprehension with no `Semaphore` → warn.

Strictness is configurable (`lint_strictness: strict | balanced | off`). The gate calibration is honest: only genuinely-unbounded constructs block; cap-absences that have a framework default raise a WARN. Findings merge into an analyzable result (adding the cap/token/fan-out categories the graph detectors don't own) or, for a recognized-but-ungraphable file, produce a cost-free lint-only result so it still gates instead of being dropped.

### 9.9 Self-healing outputs (tamper-evident verdict)

Because analysis is fully deterministic, the verdict is a pure function of (analyzed file contents + gate-affecting config + tool version). Every report therefore carries a **fingerprint** — a SHA-256 over those inputs plus a canonical digest of the verdict (gate, scores, finding categories/severities; sampling-dependent token projections are excluded so it stays reproducible). `tollgate verify <report.json> <paths>` re-derives the gate and compares: a match confirms the report reflects the code; a mismatch means the report was hand-edited, the inputs changed, or the tool version differs. In CI this is the self-heal — re-running `analyze` overwrites a stale/edited report with the canonical truth, and `verify` is the cheap step that fails the build when someone overwrites the gate output. The gate is never auto-tuned from overrides (see §13 on the feedback loop).

---

## 10. GitHub / CI integration (capability 8)

**What's built (OSS):** a **GitHub composite Action** (`action.yml`) + a **CLI** (`tollgate analyze`) + **CI templates** for GitHub Actions, GitLab CI (Code Quality report + pipeline gate), and pre-commit, plus one-shot read-only repo scanners (`scripts/scan-github-repo.sh`). The Action checks out the repo, runs the analyzer, posts a sticky PR comment, uploads **SARIF** (inline annotations on the offending file/line), and fails the check when the gate is `block`; making it a required status check blocks the merge. Output formats: terminal/markdown/json/sarif/gitlab/html, each carrying the §9.9 fingerprint.

**Platform vision (hosted GitHub App):** the rest of this section describes the hosted multi-tenant App that wraps the same analysis behind a `/analyses` API.

Delivered as a **GitHub App** with `checks:write`, `pull_requests:write`, `contents:read`.

Flow:

1. PR opened/updated → GitHub sends `pull_request` webhook to `github-app`.
2. App resolves tenant/project from installation + repo, diffs changed files for workflow/prompt artifacts, and calls `POST /analyses` with `trigger=pr`. A check-run is created immediately as `in_progress`.
3. Pipeline runs async. The static path (parse + detectors + prediction) targets <8s; full simulation runs if the diff touches behavior and config allows (<45s budget).
4. On `risk.scored`, the app updates the check-run:
   - `gate_decision=pass` → check **success**.
   - `warn` → **neutral** with annotations.
   - `block` → **failure** (blocks merge if branch protection requires the check).
5. The app posts a PR comment: risk score + band, top drivers, projected monthly **tokens** (p50/p95), and the top remediation items with **token savings**. Inline annotations attach findings to the exact file/line of the offending node where source mapping exists.

Idempotency on `(installation_id, head_sha)` so re-runs don't duplicate. Re-analysis on new commits supersedes prior check-runs. A `/tollgate recheck` PR comment command forces re-evaluation.

CI-agnostic fallback: a CLI (`tollgate analyze`) and GitHub Action wrap the same `/analyses` API for GitLab/Jenkins/Buildkite users.

---

## 11. Multi-tenancy, security & isolation

- **Tenant resolution** at the gateway from API key / GitHub installation → `tenant_id` injected into request context and every event.
- **OLTP isolation:** Postgres RLS on `tenant_id` (§7.1); the pooled DB role can never read across tenants because every query is filtered by the session GUC. Defense in depth over app-layer checks.
- **OLAP isolation:** ClickHouse partitioned and ordered by `tenant_id`; query layer injects a mandatory `tenant_id =` predicate; row policies as backstop.
- **Prompt/IR confidentiality:** prompt templates and IR can contain sensitive business logic. Stored in S3 encrypted with **per-tenant KMS keys**; only the parser/prediction services hold decrypt grants. Telemetry stores **token counts and metadata, never prompt/response content** by default (opt-in sampling for debugging, redacted).
- **Region pinning:** `tenant.region` / `data_residency` route storage and processing to the tenant's region; the bus is regional with no cross-region replication of payloads.
- **AuthN/Z:** OIDC for console (SSO/SAML for enterprise), scoped API keys (argon2-hashed), GitHub App tokens. RBAC roles: owner/admin/engineer/viewer.
- **Secrets:** customer provider keys (if supplied for richer analysis) held in a vault, never logged, never in telemetry.
- **Audit log:** every policy change, gate override, and key action recorded immutably.
- **Supply chain:** event schemas pinned in a registry with backward-compat checks in CI.

---

## 12. Observability & SLOs

- **Tracing:** W3C `trace_id` flows from gateway through every event envelope; one analysis is one trace across services.
- **Metrics:** per-stage latency histograms (parser/predict/sim/score), gate decision rates, prediction calibration error (predicted p95 / observed p95), ingest lag, bus consumer lag, DLQ depth.
- **Self-metering:** the platform meters its own compute usage per tenant for internal attribution (dogfooding).
- **SLOs** per §3 NFR table; error budgets per service; alerts on calibration drift > 25% and ingest lag > 60s.

---

## 13. Validation, field study & the feedback loop *(built)*

Tollgate separates **proven correctness** from **behavior in the wild**, and has a *governed* loop for turning real findings into better detection — without ever letting the gate train itself permissive.

**Correctness — `validation/`.** A hand-labeled corpus (`corpus/labels.yaml` + cases) with known-correct answers. `harness.py` scores discovery, unbounded-loop precision/recall/F1 against trivial baselines it must beat, gate accuracy and recommendation accuracy; `--strict` makes it a CI gate. It supports a **train/held-out split** (deterministic fold assignment) so detection is never evaluated on the cases it was tuned against. Metamorphic, fuzz, mutation and determinism suites live here too.

**Behavior in the wild — the field study.** A harness clones and analyzes a large public population of agent repos and reports **honestly**: coverage, gate split, and finding histograms are *descriptive counts*, not validated-correct numbers. The single correctness claim the study may publish is an **adjudicated precision with a Wilson 95% CI** from a hand-labeled random sample (`sample.py` → `precision.py`; `recall.py` for miss-rate). An independent **auto-triage oracle** (`auto_triage.py`) re-checks the kwarg-decidable findings against source so humans only adjudicate disagreements — reported as *agreement*, never as validated precision. The study harness is internal; only its **anonymized** results (`FIELD-STUDY.md`, `docs/field-study.html`) and the named-population credits (`ACKNOWLEDGEMENTS.md`) are published.

**The feedback loop (governed, deterministic).** The hard rule: **never auto-tune detectors/thresholds against the evaluation set** — that launders a wrong answer into a confident one. Instead:

1. A human adjudicates a finding (false positive, or a miss).
2. `promote_to_fixtures.py` turns each confirmed mistake into a frozen `validation/corpus` case with corrected ground truth and provenance — feedback becomes a *regression test*, not a weight.
3. `threshold_proposals.py` surfaces severity/threshold changes from finding mix and (if a labeled sample exists) measured FP rates — as *proposals for human review*, never auto-applied.
4. A human makes the code/config change; precision/recall is re-measured on the **held-out** slice; CI (`harness.py --strict`) gates on no-regression.
5. `study/feedback.py` deterministically re-derives every published headline from the raw results and flags anything that smells like self-tuning.

The gate stays rule-based, deterministic, and human-authored; the loop makes it *more accurate over time* without ever making it permissive on demand.

---

## 14. MVP roadmap & capability coverage

Phased to ship the prevention core first; reporting/telemetry depth follows because it only sharpens an already-useful predictor. **Phases 0–2 (the deterministic pre-deploy core) are implemented in the open-source package**, extended well beyond the original P1 scope (CrewAI/AutoGPT/imperative parsers, the strict agentic linter, self-healing outputs, the validation + field-study subsystems). Phases 3–5 (hosted telemetry/SaaS planes) remain the target.

### Phase 0 — Foundations (weeks 1–3) — *partially built*
Model Catalog seeded with OpenAI + Anthropic + Gemini + open models and illustrative pricing, package scaffold, CI, test/validation baseline. ✅ *(The hosted pieces — tenant/project/auth, Postgres + RLS, event bus — are platform vision.)*

### Phase 1 — Static prevention MVP — *built, and expanded* ✅
Workflow Parser for LangGraph **plus CrewAI, AutoGPT exports, framework-agnostic imperative agents, a native DSL and raw prompt templates** → IR. Tokenizer-static + heuristic Prediction. Context-explosion and recursive/delegation-loop detectors **plus the strict agentic linter** (§9.8) reaching LangChain/AutoGen/LlamaIndex/smolagents. Risk Scorer with default gate, **self-healing fingerprint + `verify`** (§9.9). Shipped as a **CLI + GitHub Action + CI templates** (the hosted App is vision). Delivers "block a runaway loop / uncapped agent before merge."
*Exit (met):* a PR with an unbounded loop or an uncapped/ungoverned agent is blocked with an actionable, deterministic comment in <1s on the static path.

### Phase 2 — Simulation, substitution & remediation — *built* ✅
Monte Carlo Token Sim Engine + traffic scenarios (10k req/week base; per-day/week overrides), what-if model overrides, Model Substitution Recommender, Remediation plan with token-savings estimates and concrete suggested changes. Policy Engine (token_ceiling, allowlist, context_cap, loop_guard, gate_threshold) on the pre-deploy path. Plus the **validation + field-study + feedback subsystems** (§13).
*Exit (met):* report shows projected monthly tokens and "right-size X→Y (cost lever)," with a labeled-corpus benchmark gating correctness.

### Phase 3 — Telemetry & recalibration (weeks 14–19) — *platform vision*
Telemetry ingest (OTLP-JSON), ClickHouse warehouse + hourly MV, drift detection, telemetry-backed Prediction basis, runtime policy evaluation. Prediction calibration target ±20%.
*Exit:* predictions recalibrate from production; runtime token-ceiling/loop violations fire.

### Phase 4 — Exec forecasting & breadth (weeks 20–26)
Forecast Service (exec dashboard, driver decomposition, blended projected+actual), CrewAI + OpenAI Assistants parsers, OSS self-hosted cost modeling, SSO/SAML, webhooks, audit log.
*Exit:* CFO-facing forecast with remediation-linked savings opportunities.

### Phase 5 — Scale & enterprise (weeks 27+)
Multi-region/residency, 100k events/s ingest hardening, advanced detectors (fan-out, retry storms), policy-as-code bundles, GitLab/Jenkins adapters, SOC 2 controls.

### Capability coverage map

| # | Capability | Status | Primary component |
|---|---|---|---|
| 1 | Parse agent workflows/prompts (LangGraph, CrewAI, AutoGPT, imperative, DSL, prompts) | ✅ built | `parsers/` → Workflow IR |
| 2 | Predict token consumption (static/heuristic; telemetry-backed is vision) | ✅ built (static) · 🔜 telemetry | `prediction.py` |
| 3 | Simulate token usage under traffic | ✅ built | `simulation.py` |
| 4 | Detect context explosion | ✅ built | `detectors.py` |
| 5 | Detect recursive/delegation loops | ✅ built | `detectors.py` + `graphutil.py` |
| 6 | Recommend cheaper models (cost lever) | ✅ built | `substitution.py` |
| 7 | Deployment risk score + gate | ✅ built | `scoring.py` |
| 8 | GitHub/GitLab CI checks (Action + CLI + templates; hosted App is vision) | ✅ built · 🔜 hosted App | `action.yml`, `cli.py`, `report.py` |
| 9 | OpenAI/Anthropic/Gemini/open models | ✅ built | `catalog.py`, `data/models.yaml` |
| 10 | Exec forecasts + engineering remediation | ✅ built | `forecast.py`, `remediation.py` |
| 11 | **Strict agentic lint** (unbounded loops, missing caps, uncapped output, fan-out) | ✅ built | `agentic_lint.py` |
| 12 | **Self-healing outputs** (fingerprint + `verify`) | ✅ built | `pipeline.py`, `cli.py` |
| 13 | **Validation + field study + governed feedback loop** | ✅ built | `validation/`, field-study harness |

---

*End of design v0.2. The deterministic pre-deploy core (capabilities 1–13) is implemented as the open-source `tollgate` package. Open questions for the platform planes: (a) Monte-Carlo trial count vs. latency budget on a hosted synchronous PR path; (b) inline LLM proxy for zero-instrumentation telemetry vs. SDK-only ingestion; (c) self-hosted open-model cost granularity (per-GPU-hour amortization vs. token-equivalent pricing).*

