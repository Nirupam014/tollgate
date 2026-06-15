"""HTML reporter — a self-contained visual dashboard rendered from a RunResult.

This is a first-class output format (``-f html`` / ``-o html=dashboard.html``),
so the dashboard is produced in the same ``tollgate analyze`` run as the json /
markdown / sarif reports. The data is derived from ``run.to_dict()`` — the same
payload the json reporter emits — so the dashboard never drifts from the report.

The output is a single, fully self-contained file with no build step and **no
external resources**: the analysis payload is inlined as a JSON blob the page
reads on load, and the bar charts are rendered with plain inline HTML/CSS (no
charting library, no CDN), so the report renders identically offline, in a
sandboxed file preview, and in any CI artifact viewer.

Because this runs *inside* ``tollgate analyze`` (before any scan-script cleanup),
the scanned source files still exist on disk, so we resolve exact line numbers for
findings whose node lives in a JSON/YAML graph that carries no line of its own.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .pipeline import RunResult
from .scoring import BANDS, SEVERITY_WEIGHT

_SEV_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _row_name(result: Dict[str, Any]) -> str:
    base = os.path.basename(result.get("source_path") or "")
    return base or result.get("workflow_id") or "workflow"


def _rel_path(p: Optional[str]) -> str:
    """Repo-relative path: strip the throwaway clone prefix up to '/repo/'."""
    if not p:
        return ""
    marker = "/repo/"
    i = p.rfind(marker)
    if i != -1:
        return p[i + len(marker):]
    return os.path.basename(p)


def _resolve_line(source_path: Optional[str], node_id: Optional[str],
                  line: Optional[int]) -> Optional[int]:
    """Use the finding's own line if present; else locate the node id in the file."""
    if line:
        return line
    if not source_path or not node_id:
        return None
    try:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, ln in enumerate(fh, 1):
                if node_id in ln:
                    return i
    except OSError:
        return None
    return None


def _rem_index(result: Dict[str, Any]) -> Dict[Tuple[str, Optional[str]], Dict[str, Any]]:
    """Map (category, node_id) -> remediation item for matching findings to fixes."""
    idx: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for it in ((result.get("remediation") or {}).get("items") or []):
        idx.setdefault((it.get("category"), it.get("node_id")), it)
        idx.setdefault((it.get("category"), None), it)  # category-only fallback
    return idx


def _score_definition() -> Dict[str, Any]:
    """Static, accurate description of how the risk score & gate are derived."""
    return {
        "summary": ("A 0–100 structural risk score for token waste — unbounded loops, "
                    "context explosion, fan-out, prompt bloat, retry storms. It measures "
                    "uncapped token exposure, not a dollar budget."),
        "weights": dict(SEVERITY_WEIGHT),
        "bands": [list(b) for b in BANDS],
        "rules": [
            "Within each category, finding weights are summed with diminishing returns "
            "(each additional finding counts ×0.5), so one extra low finding can't dominate.",
            "Any hard policy violation adds 100, forcing a block.",
            "Gate = BLOCK if there is any critical finding, the score ≥ 75, or a policy "
            "violation; WARN if the score ≥ 50; otherwise PASS. (Thresholds are configurable.)",
        ],
    }


def _why_works(rec: Dict[str, Any]) -> Dict[str, str]:
    """Human-readable rationale: what the current model does here, and why the
    cheaper one still covers it. Grounded in the substitution's capability score."""
    cap = float(rec.get("capability_score") or 0)
    frm = rec.get("from_model") or "the current model"
    to = rec.get("to_model") or "the cheaper model"
    node = rec.get("node_id") or "this node"
    pct = int(round(cap * 100))
    does = (f"Node '{node}' runs on {frm}, a higher-tier model. You pay for that "
            f"headroom on every call, whether or not the step actually needs "
            f"frontier-level reasoning.")
    if cap >= 0.85:
        still = (f"{to} retains ~{pct}% of {frm}'s capability on this class of task "
                 f"(capability {cap:.2f}). At that level the output quality on this "
                 f"step should be effectively unchanged — the swap is low-risk.")
    elif cap >= 0.75:
        still = (f"{to} retains ~{pct}% of {frm}'s capability (capability {cap:.2f}), "
                 f"comfortably above the configured min_capability gate. It should "
                 f"handle this step's prompts with no meaningful quality loss; sample "
                 f"a little real traffic to confirm before full rollout.")
    else:
        still = (f"{to} retains ~{pct}% of {frm}'s capability (capability {cap:.2f}). "
                 f"It clears the substitution threshold but is a closer call — A/B a "
                 f"sample of production traffic against {frm} before switching.")
    return {"does": does, "still": still}


def _scenarios(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The traffic scenarios this run projected over (from the first workflow that
    carries a simulation — all workflows share the same configured scenarios)."""
    for r in results:
        scs = ((r.get("simulation") or {}).get("scenarios")) or []
        if scs:
            return [{
                "name": s.get("name"),
                "rps": s.get("rps"),
                "horizon_days": s.get("horizon_days"),
                "peak_mult": s.get("diurnal_peak_multiplier"),
            } for s in scs]
    return []


def build_dashboard_data(run: RunResult) -> Dict[str, Any]:
    """Reduce the full run payload to the compact shape the dashboard renders."""
    d = run.to_dict()
    results: List[Dict[str, Any]] = d.get("results", []) or []

    models: Dict[str, int] = {}
    sev: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    finding_rows: List[Dict[str, Any]] = []
    recs_detail: List[Dict[str, Any]] = []
    prompt_rows: List[Dict[str, Any]] = []
    swap_acc: Dict[tuple, Dict[str, Any]] = {}
    gate_reasons: List[str] = []
    total_nodes = total_findings = total_recs = blocked = 0
    prompt_tokens_saved = 0

    for r in results:
        pred_nodes = ((r.get("prediction") or {}).get("nodes")) or []
        row_models: Dict[str, int] = {}
        for n in pred_nodes:
            m = n.get("model")
            if not m:
                continue
            row_models[m] = row_models.get(m, 0) + 1
            models[m] = models.get(m, 0) + 1
        nodes = len(pred_nodes)
        total_nodes += nodes

        findings = r.get("findings") or []
        recs = r.get("recommendations") or []
        total_findings += len(findings)
        total_recs += len(recs)

        risk = r.get("risk") or {}
        name = _row_name(r)
        gate = risk.get("gate_decision", "pass")
        if gate == "block":
            blocked += 1
        proj = risk.get("projected_monthly_tokens") or {}

        rows.append({
            "name": name,
            "kind": r.get("source_kind") or "",
            "nodes": nodes,
            "score": risk.get("score", 0),
            "band": risk.get("band", ""),
            "gate": gate,
            "findings": len(findings),
            "recs": len(recs),
            "models": row_models,
            "drivers": risk.get("drivers") or [],
            "reasons": risk.get("reasons") or [],
            "p50": proj.get("p50", 0),
            "p95": proj.get("p95", 0),
        })

        if gate == "block":
            for reason in (risk.get("reasons") or []):
                gate_reasons.append(f"{name}: {reason}")

        for pr in (r.get("prompt_reviews") or []):
            saved = pr.get("tokens_saved", 0) or 0
            prompt_tokens_saved += saved
            prompt_rows.append({
                "wf": name,
                "node": pr.get("node_id"),
                "original": pr.get("original", ""),
                "rewritten": pr.get("rewritten", ""),
                "recommendation": pr.get("recommendation", ""),
                "issues": [i.get("message", "") for i in (pr.get("issues") or [])],
                "saved": saved,
                "savings_pct": pr.get("savings_pct", 0),
                "before_tok": pr.get("original_tokens", 0),
                "after_tok": pr.get("rewritten_tokens", 0),
            })

        rem_idx = _rem_index(r)
        for f in findings:
            s = f.get("severity", "")
            sev[s] = sev.get(s, 0) + 1
            ev = f.get("evidence") or {}
            cyc = ev.get("cycle") or []
            node_id = f.get("node_id")
            rem = rem_idx.get((f.get("category"), node_id)) or rem_idx.get((f.get("category"), None))
            finding_rows.append({
                "wf": name,
                "file": _rel_path(f.get("source_path") or r.get("source_path")),
                "line": _resolve_line(f.get("source_path") or r.get("source_path"),
                                      node_id, f.get("line")),
                "node": node_id,
                "cat": f.get("category", ""),
                "sev": s,
                "cycle": cyc,
                "guard": ev.get("termination_guard") or "—",
                "iters": ev.get("estimated_iterations") or 0,
                "message": f.get("message", ""),
                "rem": ({
                    "title": rem.get("title"),
                    "detail": rem.get("detail"),
                    "suggested_change": rem.get("suggested_change"),
                    "effort": rem.get("effort"),
                } if rem else None),
            })

        for rec in recs:
            frm, to = rec.get("from_model"), rec.get("to_model")
            key = (frm, to)
            sw = swap_acc.setdefault(key, {"from": frm, "to": to, "count": 0, "sum": 0.0})
            sw["count"] += 1
            sw["sum"] += float(rec.get("savings_pct") or 0)
            recs_detail.append({
                "swap": f"{frm} → {to}",
                "wf": name,
                "file": _rel_path(r.get("source_path")),
                "line": _resolve_line(r.get("source_path"), rec.get("node_id"), None),
                "node": rec.get("node_id"),
                "from": frm,
                "to": to,
                "savings_pct": rec.get("savings_pct", 0),
                "capability": rec.get("capability_score", 0),
                "expected_calls": rec.get("expected_calls", 1),
                "current_usd": rec.get("current_call_usd"),
                "new_usd": rec.get("new_call_usd"),
                "notes": rec.get("notes") or "",
                "why": _why_works(rec),
            })

    swaps = []
    for sw in swap_acc.values():
        c = sw["count"] or 1
        swaps.append({"from": sw["from"], "to": sw["to"], "count": sw["count"],
                      "avg_savings": round(sw["sum"] / c, 1)})
    swaps.sort(key=lambda s: s["count"], reverse=True)

    # Inefficiencies the reviewer found in code/config-embedded prompts.
    for p in (d.get("detected_prompts") or []):
        pr = p.get("review")
        if not pr:
            continue
        saved = pr.get("tokens_saved", 0) or 0
        prompt_tokens_saved += saved
        loc = (p.get("source_path", "").split("/")[-1] or "?")
        prompt_rows.append({
            "wf": f"{loc} (embedded)",
            "node": p.get("name") or "prompt",
            "original": pr.get("original", ""),
            "rewritten": pr.get("rewritten", ""),
            "recommendation": pr.get("recommendation", ""),
            "issues": [i.get("message", "") for i in (pr.get("issues") or [])],
            "saved": saved,
            "savings_pct": pr.get("savings_pct", 0),
            "before_tok": pr.get("original_tokens", 0),
            "after_tok": pr.get("rewritten_tokens", 0),
        })

    rows.sort(key=lambda x: (x["score"], x["findings"], x["nodes"]), reverse=True)
    finding_rows.sort(key=lambda f: (_SEV_RANK.get(f["sev"], 0), len(f["cycle"])), reverse=True)
    prompt_rows.sort(key=lambda p: p["saved"], reverse=True)

    # Score breakdown for the info popover = the worst workflow's drivers.
    worst = rows[0] if rows else None
    max_breakdown = None
    if worst:
        max_breakdown = {
            "name": worst["name"],
            "score": worst["score"],
            "band": worst["band"],
            "drivers": worst["drivers"],
            "reasons": worst["reasons"],
        }

    # Dedupe gate reasons while preserving order.
    seen = set()
    gate_reasons = [r for r in gate_reasons if not (r in seen or seen.add(r))]

    return {
        "gate": d.get("gate_decision", "pass"),
        "gate_reasons": gate_reasons,
        "fingerprint": d.get("fingerprint"),
        "max_score": d.get("max_score", 0),
        "max_breakdown": max_breakdown,
        "score_def": _score_definition(),
        "workflows": d.get("workflow_count", len(results)),
        "scenarios": _scenarios(results),
        "blocked": blocked,
        "total_nodes": total_nodes,
        "total_findings": total_findings,
        "total_recs": total_recs,
        "sev": sev,
        "models": models,
        "swaps": swaps,
        "recs": recs_detail,
        "rows": rows,
        "findings": finding_rows,
        "prompt_reviews": prompt_rows,
        "prompt_tokens_saved": prompt_tokens_saved,
        "detected_prompts": d.get("detected_prompts", []),
        "baseline_diff": d.get("baseline_diff"),
    }


def to_html(run: RunResult) -> str:
    data = build_dashboard_data(run)
    return (_TEMPLATE
            .replace("__BASE_CSS__", BASE_CSS)
            .replace("__TOLLGATE_DATA__", json.dumps(data)))


# Shared light theme. The dashboard and the field-study report (study/report.py)
# both inline this so the two surfaces stay visually identical — one source of
# truth for colors, cards, tables, popovers, and the contained-callout pattern.
BASE_CSS = r"""
  :root{
    --bg:#fbfcff; --panel:#ffffff; --panel2:#f7f9fd; --line:#e6eaf2;
    --line2:#eef2f7; --text:#0f1b2d; --muted:#5d6b82; --faint:#8593a8;
    --accent:#2563eb; --accent-soft:#eef3ff; --good:#0f9d6e; --good-soft:#e9f8f1;
    --warn:#c77a0a; --bad:#dc2626; --bad-soft:#fdecec; --crit:#dc2626;
    --chip:#f1f5fa;
    --shadow:0 1px 2px rgba(16,30,54,.05), 0 1px 3px rgba(16,30,54,.04);
    --shadow-lg:0 12px 34px rgba(16,30,54,.14);
  }
  *{box-sizing:border-box}
  body{margin:0;color:var(--text);-webkit-font-smoothing:antialiased;
    background:var(--bg);
    background-image:radial-gradient(1200px 600px at 85% -8%, #eef4ff 0%, rgba(238,244,255,0) 60%),
                     radial-gradient(900px 500px at 0% 0%, #f3fbff 0%, rgba(243,251,255,0) 55%),
                     linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
    background-attachment:fixed;
    font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1200px;margin:0 auto;padding:34px 22px 72px}
  /* ---- header ---- */
  header{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;
    margin-bottom:26px;padding-bottom:22px;border-bottom:1px solid var(--line)}
  .brand{display:flex;align-items:center;gap:13px}
  .brand .mark{width:40px;height:40px;border-radius:10px;flex:none;
    background:linear-gradient(135deg,#2563eb,#1e40af);position:relative;box-shadow:0 2px 8px rgba(37,99,235,.28)}
  .brand .mark::before{content:"";position:absolute;inset:11px 9px;border:2px solid #fff;border-radius:2px;opacity:.95}
  .brand .mark::after{content:"";position:absolute;left:9px;right:9px;top:50%;height:2px;background:#fff;transform:translateY(-1px)}
  h1{font-size:21px;margin:0 0 3px;letter-spacing:-.2px;font-weight:700}
  .sub{color:var(--muted);font-size:12.5px}
  .gate{padding:11px 18px;border-radius:12px;font-weight:700;font-size:14px;letter-spacing:.4px;
    display:flex;align-items:center;gap:9px;border:1px solid;position:relative;box-shadow:var(--shadow)}
  .gate.block{background:var(--bad-soft);color:#b42318;border-color:#f4c5c0}
  .gate.warn{background:#fdf4e7;color:#b45309;border-color:#f3dba9}
  .gate.pass{background:var(--good-soft);color:#067a55;border-color:#b6e6d2}
  .dot{width:8px;height:8px;border-radius:50%;background:currentColor;box-shadow:0 0 0 4px rgba(0,0,0,.06)}
  /* ---- kpis ---- */
  .kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:13px;margin-bottom:26px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px 16px;
    position:relative;box-shadow:var(--shadow)}
  .kpi .v{font-size:25px;font-weight:700;line-height:1.1;letter-spacing:-.5px}
  .kpi .l{color:var(--faint);font-size:11px;text-transform:uppercase;letter-spacing:.6px;margin-top:7px;
    display:flex;align-items:center;gap:6px;font-weight:600}
  .kpi.bad .v{color:var(--bad)} .kpi.good .v{color:var(--good)} .kpi.accent .v{color:var(--accent)}
  /* ---- cards / sections ---- */
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 20px 18px;
    box-shadow:var(--shadow)}
  .card h2{font-size:13.5px;margin:0 0 16px;color:var(--text);letter-spacing:-.1px;font-weight:700;
    display:flex;align-items:center;gap:9px;padding-bottom:12px;border-bottom:1px solid var(--line2)}
  .card h2 .badge{font-size:11px;color:var(--muted);font-weight:600;background:var(--chip);
    padding:3px 10px;border-radius:20px;border:1px solid var(--line);margin-left:auto}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;color:var(--faint);font-weight:600;font-size:10.5px;text-transform:uppercase;
    letter-spacing:.6px;padding:9px 10px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);z-index:1}
  td{padding:10px;border-bottom:1px solid var(--line2);vertical-align:top}
  tr:last-child td{border-bottom:none}
  tbody tr.row:hover{background:#f5f8ff}
  tr.clk{cursor:pointer}
  .tag{display:inline-block;padding:2px 8px;border-radius:7px;font-size:11px;font-weight:600;background:var(--chip);
    border:1px solid var(--line);color:#334155;margin:1px 3px 1px 0;white-space:nowrap}
  .sev{padding:3px 9px;border-radius:7px;font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
  .sev.critical{background:var(--bad-soft);color:#b42318}
  .sev.high{background:#fdf0e6;color:#c2410c}
  .sev.medium{background:#fdf4e7;color:#b45309}
  .sev.low{background:#eef2f8;color:#475569}
  .pill{padding:3px 10px;border-radius:20px;font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
  .pill.block{background:var(--bad-soft);color:#b42318}
  .pill.warn{background:#fdf4e7;color:#b45309}
  .pill.pass{background:var(--good-soft);color:#067a55}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--muted)}
  .loc{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--accent)}
  .full{grid-column:1 / -1}
  .scroll{max-height:460px;overflow:auto;border:1px solid var(--line2);border-radius:10px}
  .scroll table{font-size:13px}
  .scroll th{padding-left:12px;padding-right:12px}
  .scroll td{padding-left:12px;padding-right:12px}
  /* contained helper callout (was free-floating gray text) */
  .note{color:var(--muted);font-size:12px;margin-top:14px;line-height:1.6;
    background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--accent);
    border-radius:0 9px 9px 0;padding:11px 14px;position:relative}
  .note::before{content:"i";position:absolute;left:-9px;top:11px;width:15px;height:15px;border-radius:50%;
    background:var(--accent);color:#fff;font-size:10px;font-weight:700;font-style:normal;
    display:flex;align-items:center;justify-content:center;font-family:Georgia,serif}
  .note b{color:var(--text);font-weight:600}
  /* ---- PR-delta banner ---- */
  .delta{border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin-bottom:22px;
    box-shadow:var(--shadow);background:var(--panel)}
  .delta.block{border-left:5px solid var(--bad);background:var(--bad-soft)}
  .delta.warn{border-left:5px solid var(--warn);background:#fdf4e7}
  .delta.pass{border-left:5px solid var(--good);background:var(--good-soft)}
  .delta .dh{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-weight:700;font-size:15px}
  .delta .counts{display:flex;gap:8px;margin:12px 0 4px;flex-wrap:wrap}
  .delta .ct{padding:4px 11px;border-radius:20px;font-size:12px;font-weight:600;border:1px solid var(--line);background:#fff}
  .delta .ct.n{color:#b42318} .delta .ct.w{color:#b45309} .delta .ct.f{color:#067a55} .delta .ct.u{color:var(--muted)}
  .delta table{margin-top:10px;background:#fff;border-radius:10px;overflow:hidden}
  .delta .sub{color:var(--muted);font-size:12px;margin-top:6px}
  .barwrap{min-height:120px;display:flex;flex-direction:column;justify-content:center}
  /* dependency-free horizontal bars (replaces the old canvas charts) */
  .hbar{display:grid;grid-template-columns:130px 1fr 96px;align-items:center;gap:10px;margin:7px 0;font-size:12px}
  .hbar-l{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .hbar-t{background:var(--line2);border-radius:6px;height:16px;overflow:hidden}
  .hbar-f{height:100%;border-radius:6px;min-width:2px;transition:width .2s}
  .hbar-v{text-align:right;font-weight:600;color:var(--text);white-space:nowrap}
  .empty{color:var(--faint);font-size:13px;padding:22px 6px;text-align:center}
  .caret{display:inline-block;width:12px;color:var(--faint);transition:transform .15s}
  tr.open .caret{transform:rotate(90deg);color:var(--accent)}
  .detail{background:var(--panel2)}
  .detail .inner{padding:6px 8px 14px 28px}
  .kv{display:grid;grid-template-columns:140px 1fr;gap:5px 14px;font-size:12.5px;margin:6px 0}
  .kv .k{color:var(--faint);font-weight:600}
  pre.code{background:#f7f9fc;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0 0;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#1f2937;white-space:pre-wrap;overflow:auto}
  .fix{margin-top:8px;padding:10px 13px;border-left:3px solid var(--good);background:var(--good-soft);
    border-radius:0 9px 9px 0;font-size:12.5px}
  .fix b{color:var(--good)}
  /* info icon + popover */
  .info{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;
    border:1px solid var(--faint);color:var(--faint);font-size:10px;font-weight:700;cursor:help;position:relative;font-style:normal}
  .info:hover{color:var(--accent);border-color:var(--accent)}
  .pop{display:none;position:absolute;z-index:30;width:320px;background:#fff;border:1px solid var(--line);
    border-radius:11px;padding:14px 15px;box-shadow:var(--shadow-lg);font-weight:400;
    color:var(--text);font-size:12.5px;line-height:1.55;text-transform:none;letter-spacing:0;left:0;top:22px}
  .info:hover .pop, .gate:hover .pop{display:block}
  .gate .pop{left:auto;right:0;top:48px;width:340px}
  .pop h4{margin:0 0 8px;font-size:11px;color:var(--accent);text-transform:uppercase;letter-spacing:.6px;font-weight:700}
  .pop ul{margin:6px 0 0;padding-left:16px}
  .pop li{margin:3px 0}
  .pop .row2{display:flex;justify-content:space-between;gap:10px;padding:3px 0;border-bottom:1px dashed var(--line)}
  .pop .row2:last-child{border-bottom:none}
  .tip{position:relative;cursor:help;border-bottom:1px dotted var(--faint)}
  /* Open the per-workflow score popover to the LEFT (score sits at its top-right
     corner) so it never widens the table or triggers a horizontal scrollbar. */
  .tip .pop{width:320px;left:auto;right:0;top:22px}
  .tip:hover .pop{display:block}
  /* The per-workflow table must NOT clip its hover popovers, so it forgoes the
     max-height scroll the other (expandable) tables use. */
  .wfscroll{overflow:visible;max-height:none;border:1px solid var(--line2);border-radius:10px}
  .wfscroll th{position:static}
  footer{color:var(--faint);font-size:11px;margin-top:34px;padding-top:16px;line-height:1.6;
    border-top:1px solid var(--line);max-width:920px}
  footer .brandline{margin-top:7px;color:var(--faint);font-weight:500}
  footer .mono{color:var(--faint)}
  footer b{color:var(--muted);font-weight:600}
  @media(max-width:900px){.kpis{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}}
"""


# The template carries a literal "__TOLLGATE_DATA__" token where the JSON blob is
# injected. Everything else is static. Kept as a plain string (not an f-string) so
# CSS/JS braces and backticks pass through untouched.
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Tollgate Scan Report</title>
<style>
__BASE_CSS__
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="mark"></div>
      <div>
        <h1>Tollgate &mdash; Token-Risk Report</h1>
        <div class="sub">Prevention-first static scan &middot; <span id="meta"></span></div>
      </div>
    </div>
    <div id="gateBadge"></div>
  </header>

  <div id="deltaBanner"></div>

  <div class="kpis" id="kpis"></div>

  <div class="grid">
    <div class="card">
      <h2>Model mix <span class="badge" id="modelCount"></span></h2>
      <div class="barwrap" id="modelBars"></div>
    </div>
    <div class="card">
      <h2>Model right-sizing (cost lever) <span class="badge" id="recCount"></span></h2>
      <div class="barwrap" id="recBars"></div>
      <div class="note" id="recNote"></div>
    </div>
  </div>

  <div class="card full" style="margin-bottom:18px">
    <h2>Critical findings <span class="badge" id="findCount"></span></h2>
    <div class="scroll"><table id="findTable">
      <thead><tr><th style="width:18px"></th><th>Location</th><th>Type</th><th>Severity</th><th>Cycle</th><th>Guard</th><th>Iters</th></tr></thead>
      <tbody></tbody></table></div>
    <div class="note">Click a finding to see the exact file/line and the recommended remediation. A cycle with <b>no termination guard</b> can run unbounded under adverse inputs &mdash; token cost is uncapped. Runtime guards (counter blocks) are not visible to a static scan and warrant human review.</div>
  </div>

  <div class="card full" style="margin-bottom:18px">
    <h2>Model right-sizing (cost lever) &mdash; detail <span class="badge" id="recDetailCount"></span></h2>
    <div class="scroll"><table id="recTable">
      <thead><tr><th style="width:18px"></th><th>Substitution</th><th>Nodes</th><th>Avg savings</th><th>Capability</th></tr></thead>
      <tbody></tbody></table></div>
    <div class="note">Click a substitution to see every workflow, node, and file path where the swap applies, with the exact change to make. Right-sizing keeps the same token volume on a cheaper model &mdash; capability scores estimate how safe each swap is.</div>
  </div>

  <div class="card full" style="margin-bottom:18px">
    <h2>Prompt token optimisation <span class="badge" id="promptCount"></span></h2>
    <div class="scroll"><table id="promptTable">
      <thead><tr><th style="width:18px"></th><th>Prompt</th><th>Current prompt</th><th>Recommendation</th><th>Example (efficient rewrite)</th><th>Saved</th></tr></thead>
      <tbody></tbody></table></div>
    <div class="note">Rewrites are produced by <b>deterministic static rules</b> &mdash; no model call, so the &ldquo;after&rdquo; is always a faithful rewrite of the &ldquo;before.&rdquo; Click a row for the full before/after and the list of issues. Review before adopting; token counts are per call.</div>
  </div>

  <div class="card full" id="detectedPromptsCard" style="margin-bottom:18px;display:none">
    <h2>Prompts detected in code/config <span class="badge" id="dpCount"></span></h2>
    <div class="scroll"><table id="dpTable">
      <thead><tr><th>Location</th><th>Name</th><th>~tokens</th><th>Why flagged</th></tr></thead>
      <tbody></tbody></table></div>
    <div class="note"><b>Heuristically detected</b> string literals (any language) that look like LLM prompts &mdash; e.g. constants in <span class="mono">prompts.py</span>/<span class="mono">prompts.ts</span> or YAML config. Surfaced for prompt-bloat / injection review; <b>not part of the risk gate</b>.</div>
  </div>

  <div class="card full">
    <h2>Per-workflow breakdown <span class="badge" id="wfCount"></span></h2>
    <div class="wfscroll"><table id="wfTable">
      <thead><tr><th>Workflow</th><th>Kind</th><th>LLM nodes</th><th>Models</th><th>Findings</th><th>Recs</th><th>Score</th><th>Gate</th></tr></thead>
      <tbody></tbody></table></div>
    <div class="note" id="wfNote">Hover a score to see how it was computed (severity-weighted drivers) and a one-line summary.</div>
  </div>

  <footer>
    <div id="scenarioNote" style="margin-bottom:8px"></div>
    Token projections are consumption estimates over the traffic scenarios above; they are not dollar figures. Edit the <span class="mono">scenarios:</span> block in your <span class="mono">.tollgate.yml</span> (or run <span class="mono">tollgate init</span> to scaffold one) to change them. Model-substitution savings use illustrative catalog prices &mdash; pass <span class="mono">--models &lt;catalog.yml&gt;</span> for your real rates.
    <div class="brandline">Generated by Tollgate &middot; prevention-first token-risk analysis for AI agents. Recommendations are for human review &mdash; the scanner modifies nothing.<span id="fpLine"></span></div>
  </footer>
</div>

<script>
const D = __TOLLGATE_DATA__;
const esc = s => String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
if (D.fingerprint) { const _fp=document.getElementById('fpLine'); if(_fp) _fp.innerHTML = ' &middot; Fingerprint <span class="mono">'+esc(D.fingerprint.slice(0,16))+'…</span> (re-derivable with <span class="mono">tollgate verify</span>)'; }
(function(){
  const dp = D.detected_prompts || [];
  if (!dp.length) return;
  const card = document.getElementById('detectedPromptsCard');
  card.style.display = '';
  document.getElementById('dpCount').textContent = dp.length;
  const tot = dp.reduce((a,p)=>a+(p.est_tokens||0),0);
  const tb = card.querySelector('#dpTable tbody');
  tb.innerHTML = dp.slice(0,200).map(p=>{
    const loc = (p.source_path? p.source_path.split('/').pop():'?') + (p.line? (':'+p.line):'');
    return '<tr><td class="mono">'+esc(loc)+'</td><td class="mono">'+esc(p.name||'—')+
           '</td><td>'+(p.est_tokens||0).toLocaleString()+'</td><td>'+esc((p.reasons||[]).slice(0,3).join(', '))+'</td></tr>';
  }).join('');
  document.getElementById('dpCount').textContent = dp.length + ' · ~' + tot.toLocaleString() + ' tokens';
})();
// --- PR-delta banner (only when run against a baseline) ----------------------
(function(){
  const bd = D.baseline_diff;
  if (!bd) return;
  const el = document.getElementById('deltaBanner');
  const g = bd.delta_gate || 'pass';
  const c = bd.counts || {};
  const dloc = f => {
    const p = (f.source_path||'').split('/').pop();
    const sfx = f.line ? (':'+f.line) : (f.node_id ? (' @'+f.node_id) : '');
    return p ? (p+sfx) : (sfx||'—');
  };
  const newRows = (bd.new||[]).slice(0,20).map(f=>{
    const occ = (f.occurrences && f.occurrences>1) ? (' ×'+f.occurrences) : '';
    return '<tr><td><span class="sev '+esc(f.severity)+'">'+esc(f.severity)+occ+'</span></td>'+
      '<td><span class="tag">'+esc(f.category)+'</span></td><td class="loc">'+esc(dloc(f))+
      '</td><td>'+esc(f.message)+'</td></tr>';
  }).join('');
  const worRows = (bd.worsened||[]).slice(0,20).map(f=>
    '<tr><td>'+esc(f.from_severity)+' → <b>'+esc(f.to_severity)+'</b></td>'+
    '<td><span class="tag">'+esc(f.category)+'</span></td><td class="loc">'+esc(dloc(f))+
    '</td><td>'+esc(f.message)+'</td></tr>').join('');
  let html = '<div class="delta '+g+'">'+
    '<div class="dh"><span class="dot"></span>Tollgate PR check: '+g.toUpperCase()+
    '<span class="sub" style="margin-left:auto">gates on the change only — pre-existing issues never fail this check</span></div>'+
    '<div class="counts"><span class="ct n">'+(c.new||0)+' new</span>'+
    '<span class="ct w">'+(c.worsened||0)+' worsened</span>'+
    '<span class="ct f">'+(c.fixed||0)+' fixed</span>'+
    '<span class="ct u">'+(c.unchanged||0)+' unchanged</span></div>';
  if (newRows) html += '<table><thead><tr><th>Severity</th><th>Type</th><th>Location</th><th>New finding</th></tr></thead><tbody>'+newRows+'</tbody></table>';
  if (worRows) html += '<table><thead><tr><th>Change</th><th>Type</th><th>Location</th><th>Worsened finding</th></tr></thead><tbody>'+worRows+'</tbody></table>';
  html += '<div class="sub">Repo-wide gate (for context): <b>'+esc((bd.full_gate||'').toUpperCase())+'</b>. '+
    'Baseline '+esc((bd.baseline_fingerprint||'').slice(0,12))+'…</div></div>';
  el.innerHTML = html;
})();
const fmtBig = n => { n=Number(n||0); return n>=1e9?(n/1e9).toFixed(1)+'B':n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':(''+Math.round(n)); };
const loc = (f,l) => f ? (esc(f)+(l?(':'+l):'')) : '—';

document.getElementById('meta').textContent = D.workflows + ' workflows analysed';

// --- Traffic scenarios used for this run (#1) ---------------------------------
(function(){
  const sc = D.scenarios || [];
  const el = document.getElementById('scenarioNote');
  if(!sc.length){ el.innerHTML = ''; return; }
  const vol = rps => {
    const perWk = Math.round(rps*604800), perDay = Math.round(rps*86400);
    if (perWk >= 1000) return '~'+perWk.toLocaleString()+' req/week';
    if (perDay >= 1)   return '~'+perDay.toLocaleString()+' req/day';
    return '~'+Math.round(rps*2592000).toLocaleString()+' req/month';
  };
  const parts = sc.map(s=>{
    const peak = s.peak_mult ? (' · '+s.peak_mult+'× diurnal peak') : '';
    return esc(s.name)+' ('+vol(Number(s.rps))+' over '+esc(s.horizon_days)+' days'+peak+')';
  }).join(' &nbsp;·&nbsp; ');
  el.innerHTML = '<b>Traffic scenarios:</b> '+parts+'.';
})();

// --- Gate badge (single source of truth) + "why blocked" popover (#1) ---------
const gb = document.getElementById('gateBadge');
gb.className = 'gate ' + D.gate;
let why = '';
if(D.gate_reasons && D.gate_reasons.length){
  why = '<div class="pop"><h4>Why the gate is '+esc(D.gate)+'</h4><ul>'+
    D.gate_reasons.slice(0,12).map(r=>'<li>'+esc(r)+'</li>').join('')+'</ul></div>';
} else if (D.gate==='pass'){
  why = '<div class="pop"><h4>Gate: pass</h4>No blocking findings or policy violations.</div>';
}
gb.innerHTML = '<span class="dot"></span> GATE: ' + (D.gate||'').toUpperCase() +
  (why ? ' <span class="info" tabindex="0">i'+why+'</span>' : '');

// --- KPIs (gate shown once, in the badge above) -------------------------------
const sd = D.score_def, mb = D.max_breakdown;
let scorePop = '<div class="pop"><h4>Max risk score</h4>'+esc(sd.summary)+
  '<div style="margin-top:8px"><b>Severity weights</b>'+
  Object.entries(sd.weights).map(([k,v])=>'<div class="row2"><span>'+esc(k)+'</span><span>'+v+'</span></div>').join('')+'</div>'+
  '<ul>'+sd.rules.map(r=>'<li>'+esc(r)+'</li>').join('')+'</ul>';
if(mb){
  scorePop += '<div style="margin-top:8px"><b>This run\'s max ('+mb.score+') — '+esc(mb.name)+'</b>'+
    (mb.drivers&&mb.drivers.length ? mb.drivers.map(dr=>'<div class="row2"><span>'+esc(dr.category)+'</span><span>+'+dr.contribution+'</span></div>').join('') : '<div>No driver breakdown.</div>')+'</div>';
}
scorePop += '</div>';

const kpis = [
  {v:D.max_score, l:'Max risk score', c:D.max_score>=50?'bad':'', info:scorePop},
  {v:D.workflows, l:'Workflows', c:'accent'},
  {v:D.blocked, l:'Blocked workflows', c:D.blocked?'bad':'good'},
  {v:D.total_nodes, l:'LLM nodes', c:''},
  {v:D.total_findings, l:'Critical findings', c:D.total_findings?'bad':''},
  {v:D.total_recs, l:'Cost recommendations', c:'good'},
];
document.getElementById('kpis').innerHTML = kpis.map(k=>
  `<div class="kpi ${k.c}"><div class="v">${k.v}</div><div class="l">${k.l}`+
  (k.info?` <span class="info" tabindex="0">i${k.info}</span>`:``)+`</div></div>`).join('');

document.getElementById('modelCount').textContent = Object.keys(D.models).length + ' models';
document.getElementById('recCount').textContent = D.total_recs + ' total';
document.getElementById('recDetailCount').textContent = D.swaps.length + ' substitutions';
document.getElementById('findCount').textContent = D.total_findings + ' total';
document.getElementById('wfCount').textContent = D.workflows + ' total';

const PALETTE = ['#2563eb','#0f9d6e','#d97706','#dc2626','#7c3aed','#0891b2','#ea580c','#db2777','#64748b','#16a34a'];

// Dependency-free horizontal bar chart (no charting lib / CDN). items:
//   {label, value, valLabel?, suffix?}  — bar length is value / max.
function hbars(el, items, empty){
  if(!el) return;
  if(!items.length){ el.innerHTML = '<div class="empty">'+esc(empty)+'</div>'; return; }
  const max = Math.max.apply(null, items.map(i=>i.value).concat([1]));
  el.innerHTML = items.map((it,idx)=>{
    const w = Math.max(2, Math.round((it.value/max)*100));
    const color = PALETTE[idx % PALETTE.length];
    const v = (it.valLabel!=null ? it.valLabel : it.value);
    return '<div class="hbar"><div class="hbar-l" title="'+esc(it.label)+'">'+esc(it.label)+'</div>'+
      '<div class="hbar-t"><div class="hbar-f" style="width:'+w+'%;background:'+color+'"></div></div>'+
      '<div class="hbar-v">'+esc(v)+esc(it.suffix||'')+'</div></div>';
  }).join('');
}

// Dependency-free vertical grouped bar chart (inline SVG, no lib / CDN): two
// series on dual axes — savings % (left, 0-100) and # nodes (right). Mirrors the
// original right-sizing chart so both metrics are visible per substitution.
function svgGroupedBars(el, groups, empty){
  if(!el) return;
  if(!groups.length){ el.innerHTML = '<div class="empty">'+esc(empty)+'</div>'; return; }
  const W=560,H=300,L=40,R=40,T=12,B=92, pw=W-L-R, ph=H-T-B;
  const n=groups.length, gw=pw/n, barW=Math.max(6, Math.min(24, gw*0.30));
  const leftMax=100;
  const rightMax=Math.max.apply(null, groups.map(g=>g.nodes).concat([1]));
  const rTicks=Math.max(1, Math.min(5, rightMax));
  const green='#0f9d6e', blue='#2563eb', baseY=T+ph;
  let s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" style="height:auto;max-height:300px" '+
        'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif">';
  for(let t=0;t<=100;t+=20){
    const y=baseY-(t/leftMax)*ph;
    s+='<line x1="'+L+'" y1="'+y+'" x2="'+(L+pw)+'" y2="'+y+'" stroke="#e8edf4"/>'+
       '<text x="'+(L-6)+'" y="'+(y+3)+'" font-size="9" fill="#8593a8" text-anchor="end">'+t+'</text>';
  }
  for(let i=0;i<=rTicks;i++){
    const val=Math.round(rightMax*i/rTicks), y=baseY-(val/rightMax)*ph;
    s+='<text x="'+(L+pw+6)+'" y="'+(y+3)+'" font-size="9" fill="#8593a8" text-anchor="start">'+val+'</text>';
  }
  s+='<text x="'+(L-26)+'" y="'+(T+ph/2)+'" font-size="9" fill="#8593a8" text-anchor="middle" '+
     'transform="rotate(-90 '+(L-26)+' '+(T+ph/2)+')">savings %</text>'+
     '<text x="'+(L+pw+30)+'" y="'+(T+ph/2)+'" font-size="9" fill="#8593a8" text-anchor="middle" '+
     'transform="rotate(90 '+(L+pw+30)+' '+(T+ph/2)+')"># nodes</text>';
  groups.forEach((g,i)=>{
    const cx=L+gw*(i+0.5);
    const sH=(g.savings/leftMax)*ph, nH=(g.nodes/rightMax)*ph;
    s+='<rect x="'+(cx-barW-1)+'" y="'+(baseY-sH)+'" width="'+barW+'" height="'+sH+'" rx="2" fill="'+green+'">'+
       '<title>'+esc(g.label)+': '+g.savings+'% savings</title></rect>'+
       '<rect x="'+(cx+1)+'" y="'+(baseY-nH)+'" width="'+barW+'" height="'+nH+'" rx="2" fill="'+blue+'">'+
       '<title>'+esc(g.label)+': '+g.nodes+' node(s)</title></rect>';
    const ly=baseY+12, lab=g.label.length>28?g.label.slice(0,27)+'…':g.label;
    s+='<text x="'+cx+'" y="'+ly+'" font-size="8.5" fill="#5d6b82" text-anchor="end" '+
       'transform="rotate(-25 '+cx+' '+ly+')">'+esc(lab)+'</text>';
  });
  s+='<line x1="'+L+'" y1="'+baseY+'" x2="'+(L+pw)+'" y2="'+baseY+'" stroke="#cbd5e1"/>';
  const lgY=H-12;
  s+='<rect x="'+(L+pw/2-90)+'" y="'+(lgY-9)+'" width="10" height="10" rx="2" fill="'+green+'"/>'+
     '<text x="'+(L+pw/2-75)+'" y="'+lgY+'" font-size="10" fill="#5d6b82">Avg savings %</text>'+
     '<rect x="'+(L+pw/2+20)+'" y="'+(lgY-9)+'" width="10" height="10" rx="2" fill="'+blue+'"/>'+
     '<text x="'+(L+pw/2+35)+'" y="'+lgY+'" font-size="10" fill="#5d6b82"># nodes</text>';
  s+='</svg>';
  el.innerHTML=s;
}

const mk = Object.keys(D.models);
hbars(document.getElementById('modelBars'),
  mk.map(k=>({label:k, value:D.models[k], valLabel:D.models[k],
              suffix:' node'+(D.models[k]===1?'':'s')})),
  'No LLM nodes detected.');

svgGroupedBars(document.getElementById('recBars'),
  D.swaps.map(s=>({label:s.from+' → '+s.to, savings:s.avg_savings, nodes:s.count})),
  'No cheaper substitutions recommended.');
if(D.swaps.length){
  const top = D.swaps[0];
  document.getElementById('recNote').innerHTML =
    'Largest lever: <b>'+esc(top.count)+'× '+esc(top.from)+' → '+esc(top.to)+
    '</b> at ~'+top.avg_savings+'% token-cost reduction per call. See the detail table below for exact code paths.';
} else {
  document.getElementById('recNote').innerHTML = '';
}

// --- Findings: expandable rows with exact file:line + remediation (#4) ---------
const fb = document.querySelector('#findTable tbody');
if(D.findings.length){
  D.findings.forEach((f,i)=>{
    const cyc = f.cycle && f.cycle.length;
    const tr = document.createElement('tr');
    tr.className='row clk'; tr.dataset.t='f'+i;
    tr.innerHTML = `<td><span class="caret">▶</span></td>
      <td class="loc">${loc(f.file,f.line)}${f.node?`<div class="mono">node ${esc(f.node)}</div>`:''}</td>
      <td><span class="tag">${esc(f.cat)}</span></td>
      <td><span class="sev ${esc(f.sev)}">${esc(f.sev)}</span></td>
      <td>${cyc?(f.cycle.length+' node'+(f.cycle.length>1?'s':'')+(f.cycle.length===1?' (self-loop)':'')):'—'}</td>
      <td>${f.guard==='none_detected'?'<span style="color:#dc2626;font-weight:600">none</span>':esc(f.guard)}</td>
      <td>${f.iters?('~'+f.iters):'—'}</td>`;
    const dt = document.createElement('tr');
    dt.className='detail'; dt.dataset.d='f'+i; dt.style.display='none';
    const rem = f.rem;
    dt.innerHTML = `<td colspan="7"><div class="inner">
      <div class="kv"><span class="k">File</span><span class="loc">${loc(f.file,f.line)}</span>
        <span class="k">What</span><span>${esc(f.message)}</span>
        ${cyc?`<span class="k">Cycle path</span><span class="mono">${f.cycle.map(esc).join(' → ')} → ${esc(f.cycle[0])}</span>`:''}
        ${f.cat==='recursive_loop'?`<span class="k">Termination</span><span>${f.guard==='none_detected'?'<span style="color:#dc2626;font-weight:600">no guard detected</span>':esc(f.guard)} · est. ~${f.iters||'?'} iterations</span>`:''}</div>
      ${rem?`<div class="fix"><b>Recommended remediation</b> ${rem.effort?`<span class="tag">${esc(rem.effort)} effort</span>`:''}<div style="margin:5px 0">${esc(rem.title||'')}</div>${rem.detail?`<div style="color:var(--muted)">${esc(rem.detail)}</div>`:''}${rem.suggested_change?`<pre class="code">${esc(rem.suggested_change)}</pre>`:''}</div>`:`<div class="note">No automated remediation suggestion for this finding.</div>`}
    </div></td>`;
    fb.appendChild(tr); fb.appendChild(dt);
  });
} else {
  fb.innerHTML = '<tr><td colspan="7" class="empty">No findings &mdash; nothing blocking.</td></tr>';
}

// --- Recommendations: expandable by substitution, exact code paths (#3) --------
const rb = document.querySelector('#recTable tbody');
if(D.swaps.length){
  D.swaps.forEach((s,i)=>{
    const label = s.from+' → '+s.to;
    const items = D.recs.filter(r=>r.swap===label);
    const tr = document.createElement('tr');
    tr.className='row clk'; tr.dataset.t='r'+i;
    tr.innerHTML = `<td><span class="caret">▶</span></td>
      <td><span class="tag">${esc(s.from)}</span> → <span class="tag">${esc(s.to)}</span></td>
      <td>${s.count}</td><td class="savings" style="color:var(--good);font-weight:700">~${s.avg_savings}%</td>
      <td>${items.length?items[0].capability:''}</td>`;
    const dt = document.createElement('tr');
    dt.className='detail'; dt.dataset.d='r'+i; dt.style.display='none';
    const rowsHtml = items.map(it=>`<div class="inner" style="padding-left:26px;border-bottom:1px solid var(--panel2)">
      <div class="kv">
        <span class="k">Where</span><span>In workflow <b>${esc(it.wf)}</b>, change the model on node <span class="mono">${esc(it.node)}</span></span>
        <span class="k">File</span><span class="loc">${loc(it.file,it.line)}</span>
        <span class="k">Change</span><span>${esc(it.from)} → <b style="color:var(--good)">${esc(it.to)}</b> · ~${it.savings_pct}% cheaper/call · capability ${it.capability}</span>
        ${(it.current_usd!=null&&it.new_usd!=null)?`<span class="k">Per-call cost</span><span>$${Number(it.current_usd).toFixed(5)} → $${Number(it.new_usd).toFixed(5)} (×${it.expected_calls} call${it.expected_calls==1?'':'s'}/request)</span>`:''}
        ${it.notes?`<span class="k">Note</span><span>${esc(it.notes)}</span>`:''}
      </div>
      <pre class="code">model: ${esc(it.to)}   # was ${esc(it.from)}</pre>
      ${it.why?`<div class="fix" style="border-left-color:var(--accent);background:rgba(91,157,255,.06)"><b style="color:var(--accent)">Why this still works</b><div style="margin:5px 0">${esc(it.why.does)}</div><div>${esc(it.why.still)}</div></div>`:''}
      </div>`).join('');
    dt.innerHTML = `<td colspan="5">${rowsHtml}</td>`;
    rb.appendChild(tr); rb.appendChild(dt);
  });
} else {
  rb.innerHTML = '<tr><td colspan="5" class="empty">No cheaper substitutions recommended.</td></tr>';
}

// Accordion toggling for both expandable tables.
document.querySelectorAll('tr.clk').forEach(tr=>{
  tr.addEventListener('click',()=>{
    const d = document.querySelector(`tr.detail[data-d="${tr.dataset.t}"]`);
    if(!d) return;
    const open = d.style.display!=='none';
    d.style.display = open?'none':'table-row';
    tr.classList.toggle('open', !open);
  });
});

// --- Prompt token optimisation: expandable before/after rows -----------------
const pr = D.prompt_reviews || [];
const pb = document.querySelector('#promptTable tbody');
document.getElementById('promptCount').textContent =
  pr.length ? (pr.length + ' prompt' + (pr.length>1?'s':'') + ' · ~' + fmtBig(D.prompt_tokens_saved||0) + ' tok/call reclaimable') : '0';
const snip = (s,n)=>{ s=String(s||'').replace(/\s+/g,' ').trim(); return s.length>n?esc(s.slice(0,n-1))+'…':esc(s); };
if(pr.length){
  pr.forEach((p,i)=>{
    const saved = p.saved ? (p.saved + ' tok (' + Math.round(p.savings_pct) + '%)') : 'structural';
    const tr = document.createElement('tr');
    tr.className='row clk'; tr.dataset.t='p'+i;
    tr.innerHTML = `<td><span class="caret">▶</span></td>
      <td class="mono">${esc(p.wf)}<div class="mono">node ${esc(p.node)}</div></td>
      <td>${snip(p.original,120)}</td>
      <td>${snip(p.recommendation,140)}</td>
      <td style="color:#047857">${snip(p.rewritten,120)}</td>
      <td style="color:var(--good);font-weight:700;white-space:nowrap">${esc(saved)}</td>`;
    const dt = document.createElement('tr');
    dt.className='detail'; dt.dataset.d='p'+i; dt.style.display='none';
    const issues = (p.issues&&p.issues.length)
      ? '<ul>'+p.issues.map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>'
      : '<div class="note">No issues.</div>';
    dt.innerHTML = `<td colspan="6"><div class="inner">
      <div class="kv"><span class="k">Tokens</span><span>${p.before_tok} → ${p.after_tok} (saved ${p.saved}, ${Math.round(p.savings_pct)}%)</span></div>
      <div style="margin:8px 0 4px;color:var(--faint);font-size:11px;font-weight:600;letter-spacing:.5px">CURRENT PROMPT</div>
      <pre class="code" style="color:#b42318">${esc(p.original)}</pre>
      <div style="margin:10px 0 4px;color:var(--faint);font-size:11px;font-weight:600;letter-spacing:.5px">EXAMPLE — EFFICIENT REWRITE</div>
      <pre class="code" style="color:#047857">${esc(p.rewritten)}</pre>
      <div class="fix"><b>Recommendations</b>${issues}</div>
    </div></td>`;
    pb.appendChild(tr); pb.appendChild(dt);
  });
  pb.querySelectorAll('tr.clk').forEach(tr=>{
    tr.addEventListener('click',()=>{
      const d = document.querySelector(`tr.detail[data-d="${tr.dataset.t}"]`);
      if(!d) return;
      const open = d.style.display!=='none';
      d.style.display = open?'none':'table-row';
      tr.classList.toggle('open', !open);
    });
  });
} else {
  pb.innerHTML = '<tr><td colspan="6" class="empty">No prompt inefficiencies detected (or reviewer disabled).</td></tr>';
}

// --- Per-workflow table with score-breakdown tooltip (#5) ---------------------
// Hide files with no LLM nodes AND nothing to act on — they're parsed but carry
// no token cost, no findings, and no recommendations, so they're just noise.
const gpill = g=>`<span class="pill ${g}">${g}</span>`;
const wfShown = D.rows.filter(r => r.nodes>0 || r.findings>0 || r.recs>0);
const wfHidden = D.rows.length - wfShown.length;
const wfList = wfShown.length?wfShown:D.rows;
document.getElementById('wfCount').textContent = wfList.length + ' with LLM activity';

const wfRow = (r,cls)=>{
  const tags = Object.keys(r.models).map(m=>`<span class="tag">${esc(m)}×${r.models[m]}</span>`).join('');
  const drv = (r.drivers&&r.drivers.length)
    ? r.drivers.map(d=>`<div class="row2"><span>${esc(d.category)}</span><span>+${d.contribution}</span></div>`).join('')
    : '<div>No weighted drivers (score from policy/critical gate).</div>';
  const reasons = (r.reasons&&r.reasons.length)?'<ul>'+r.reasons.map(x=>`<li>${esc(x)}</li>`).join('')+'</ul>':'';
  const pop = `<div class="pop"><h4>Score ${r.score} · band ${esc(r.band||'—')}</h4>`+
    `Severity-weighted drivers (diminishing returns per category):${drv}`+
    `<div style="margin-top:6px;color:var(--muted)">Projected ~${fmtBig(r.p50)}–${fmtBig(r.p95)} tokens/mo (p50–p95).</div>${reasons}</div>`;
  return `<tr class="row ${cls||''}"><td class="mono">${esc(r.name)}</td><td>${esc(r.kind)}</td><td>${r.nodes}</td>
    <td>${tags||'<span style="color:var(--faint)">&mdash;</span>'}</td>
    <td>${r.findings? '<b style="color:#dc2626">'+r.findings+'</b>':'0'}</td>
    <td>${r.recs||0}</td>
    <td><span class="tip"><b>${r.score}</b>${pop}</span></td>
    <td>${gpill(r.gate)}</td></tr>`;
};

// Surface the workflows that need attention (block/warn); roll the clean ones up
// into a single expandable summary so the table can't run on forever.
const wfAttention = wfList.filter(r=>r.gate!=='pass');
const wfPass = wfList.filter(r=>r.gate==='pass');
let wfHtml = wfAttention.map(r=>wfRow(r)).join('');
if(wfPass.length){
  const names = wfPass.map(p=>esc(p.name));
  const extra = names.length>2 ? (' and '+(names.length-2)+' other'+(names.length-2>1?'s':'')) : '';
  const summary = names.slice(0,2).join(', ') + extra;
  wfHtml += `<tr class="row clk wfpass-row" id="wfPassToggle"><td colspan="8" style="background:var(--panel2)">
    <span class="caret">▶</span> <span class="pill pass">pass</span>
    <b style="margin-left:4px">${wfPass.length} passing workflow${wfPass.length>1?'s':''}</b>
    <span style="color:var(--muted)"> — ${summary}. No findings or blocking risk; click to expand.</span></td></tr>`;
  wfHtml += wfPass.map(r=>wfRow(r,'wfpass')).join('');
}
if(!wfAttention.length && !wfPass.length){
  wfHtml = '<tr><td colspan="8" class="empty">No workflows with LLM activity.</td></tr>';
}
document.querySelector('#wfTable tbody').innerHTML = wfHtml;

// Passing rows start collapsed under the summary row.
document.querySelectorAll('#wfTable tr.wfpass').forEach(tr=>tr.style.display='none');
const wfToggle = document.getElementById('wfPassToggle');
if(wfToggle){
  wfToggle.addEventListener('click',()=>{
    const open = wfToggle.classList.toggle('open');
    document.querySelectorAll('#wfTable tr.wfpass').forEach(tr=>tr.style.display=open?'table-row':'none');
  });
}

let wfNoteHtml = 'Hover a score to see how it was computed (severity-weighted drivers) and a one-line summary. '+
  'Passing workflows are collapsed by default &mdash; click the <b>pass</b> row to expand them.';
if(wfHidden>0){
  wfNoteHtml += ' <b>'+wfHidden+'</b> file'+(wfHidden>1?'s were':' was')+
    ' parsed but had no LLM nodes, findings, or recommendations and '+(wfHidden>1?'are':'is')+' omitted here.';
}
document.getElementById('wfNote').innerHTML = wfNoteHtml;
</script>
</body>
</html>
"""
