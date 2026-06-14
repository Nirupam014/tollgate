"""End-to-end and unit tests for the Tollgate analyzer.

Run with: python -m unittest discover -s tests  (no third-party deps required).
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tollgate.catalog import ModelCatalog
from tollgate.config import Config
from tollgate.graphutil import find_cycles, expected_executions, component_iterations
from tollgate.ir import Guard, IREdge, IRNode, Workflow
from tollgate.parsers import discover, parse_file
from tollgate.pipeline import analyze_path, analyze_workflow
from tollgate.prediction import PredictionEngine
from tollgate.scoring import RiskScorer
from tollgate.tokenizer import count_tokens, heuristic_tokens

EXAMPLES = os.path.join(ROOT, "examples", "workflows")


def _wf_unbounded_loop():
    nodes = [IRNode("a", "llm_call", "gpt-4o", appends_history=True),
             IRNode("b", "llm_call", "gpt-4o")]
    edges = [IREdge("a", "b", "sequence"), IREdge("b", "a", "loop")]  # no guard
    return Workflow("loop_wf", "dsl", nodes, edges, entry="a")


def _wf_bounded_loop():
    nodes = [IRNode("a", "llm_call", "gpt-4o-mini"),
             IRNode("b", "llm_call", "gpt-4o-mini")]
    edges = [IREdge("a", "b", "sequence"),
             IREdge("b", "a", "loop", guard=Guard(max_depth=3))]
    return Workflow("bounded_wf", "dsl", nodes, edges, entry="a")


class TestTokenizer(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(count_tokens(""), 0)

    def test_monotonic_and_positive(self):
        short = heuristic_tokens("hello world")
        long = heuristic_tokens("hello world " * 50)
        self.assertGreater(short, 0)
        self.assertGreater(long, short)


class TestGraph(unittest.TestCase):
    def test_detect_unbounded_cycle(self):
        wf = _wf_unbounded_loop()
        cycles = find_cycles(wf)
        self.assertEqual(len(cycles), 1)
        _iters, bounded = component_iterations(wf, cycles[0])
        self.assertFalse(bounded)

    def test_bounded_cycle_iterations(self):
        wf = _wf_bounded_loop()
        cycles = find_cycles(wf)
        iters, bounded = component_iterations(wf, cycles[0])
        self.assertTrue(bounded)
        self.assertEqual(iters, 3)

    def test_expected_executions_loop_multiplies(self):
        wf = _wf_unbounded_loop()
        execs = expected_executions(wf)
        # Cyclic nodes should be expected to run more than once.
        self.assertGreater(execs["a"], 1.0)

    def test_dag_no_cycle(self):
        nodes = [IRNode("x", "llm_call", "gpt-4o-mini"), IRNode("y", "llm_call", "gpt-4o-mini")]
        edges = [IREdge("x", "y", "sequence")]
        wf = Workflow("dag", "dsl", nodes, edges, entry="x")
        self.assertEqual(find_cycles(wf), [])
        execs = expected_executions(wf)
        self.assertAlmostEqual(execs["x"], 1.0)
        self.assertAlmostEqual(execs["y"], 1.0)


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.catalog = ModelCatalog.load()

    def test_percentiles_ordered(self):
        wf = _wf_bounded_loop()
        pred = PredictionEngine(self.catalog).predict(wf)
        for n in pred.nodes:
            self.assertLessEqual(n.input_tokens.p50, n.input_tokens.p95)
            self.assertLessEqual(n.input_tokens.p95, n.input_tokens.p99)
        self.assertGreater(pred.request_cost_usd.p50, 0)

    def test_cost_increases_with_expensive_model(self):
        cheap = Workflow("c", "dsl", [IRNode("n", "llm_call", "gpt-4o-mini",
                         prompt_template="x" * 400)], [], entry="n")
        pricey = Workflow("p", "dsl", [IRNode("n", "llm_call", "claude-opus-4",
                          prompt_template="x" * 400)], [], entry="n")
        eng = PredictionEngine(self.catalog)
        # give both the same static tokens
        cheap.nodes[0].static_input_tokens = 1000
        pricey.nodes[0].static_input_tokens = 1000
        cc = eng.predict(cheap).request_cost_usd.p50
        pc = eng.predict(pricey).request_cost_usd.p50
        self.assertGreater(pc, cc)


class TestScoring(unittest.TestCase):
    def test_no_findings_passes(self):
        scorer = RiskScorer()
        rs = scorer.score([], {"p50": 0, "p95": 0})
        self.assertEqual(rs.gate_decision, "pass")
        self.assertEqual(rs.score, 0)

    def test_critical_blocks(self):
        from tollgate.findings import Finding
        scorer = RiskScorer()
        f = Finding("x", "recursive_loop", "critical", "unbounded loop")
        rs = scorer.score([f], {"p50": 0, "p95": 0})
        self.assertEqual(rs.gate_decision, "block")


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(trials=600)

    def test_runaway_agent_blocks(self):
        wf = parse_file(os.path.join(EXAMPLES, "runaway_agent.yaml"))
        res = analyze_workflow(wf, cfg=self.cfg)
        cats = {f.category for f in res.findings}
        self.assertIn("recursive_loop", cats)
        self.assertIn("context_explosion", cats)
        self.assertEqual(res.risk.gate_decision, "block")

    def test_safe_pipeline_passes_or_warns(self):
        wf = parse_file(os.path.join(EXAMPLES, "safe_pipeline.yaml"))
        res = analyze_workflow(wf, cfg=self.cfg)
        # No critical findings expected.
        self.assertFalse(any(f.severity == "critical" for f in res.findings))
        self.assertIn(res.risk.gate_decision, ("pass", "warn"))

    def test_recommendations_on_runaway(self):
        wf = parse_file(os.path.join(EXAMPLES, "runaway_agent.yaml"))
        res = analyze_workflow(wf, cfg=self.cfg)
        # Opus on routing/reasoning should yield at least one cheaper-model rec.
        self.assertTrue(len(res.recommendations) >= 1)

    def test_prompt_file_parsing(self):
        wf = parse_file(os.path.join(EXAMPLES, "summarizer.prompt"))
        self.assertEqual(len(wf.nodes), 1)
        self.assertTrue(wf.nodes[0].retrieves_context)

    def test_langgraph_parsing_finds_loop(self):
        wf = parse_file(os.path.join(ROOT, "examples", "langgraph_agent.py"))
        self.assertTrue(any(n.node_id == "plan" for n in wf.nodes))
        self.assertTrue(len(find_cycles(wf)) >= 1)

    def test_run_directory_gate(self):
        run = analyze_path([EXAMPLES], cfg=self.cfg)
        self.assertGreaterEqual(len(run.results), 2)
        self.assertEqual(run.gate_decision, "block")  # runaway dominates


class TestDiscoveryNoiseFilter(unittest.TestCase):
    """A scan must surface genuine agent artifacts only, never docs/meta noise."""

    def _tree(self):
        import tempfile
        d = tempfile.mkdtemp()

        def w(rel, body=""):
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(body)

        # Noise: documentation, project-meta, CI, license — never agents.
        w("README.md", "# Project\nSome docs. Use {x} maybe.")
        w("CONTRIBUTING.md", "Please contribute")
        w("CODE_OF_CONDUCT.md", "Be nice")
        w("AGENTS.md", "Agent guidance for humans")
        w("SKILL.md", "---\nname: foo\n---\ndocs")
        w("LICENSE", "MIT")
        w("docs/guide.md", "Lots of {{ docs }} here")        # docs/ pruned
        w(".github/workflows/ci.yml", "name: ci\non: push")  # .github pruned
        # Genuine artifacts.
        w("agents/planner.prompt", "You are a planner. Topic: {{ topic }}")
        w("flows/graph.yaml", "nodes:\n  - id: a\n    type: llm_call\n    model: gpt-4o")
        w("prompts/summary.md", "Summarize {{ doc }} in 3 lines.")
        return d

    def test_only_genuine_artifacts_discovered(self):
        d = self._tree()
        found = {os.path.relpath(p, d) for p in discover([d])}
        self.assertEqual(found, {
            "agents/planner.prompt",
            "flows/graph.yaml",
            "prompts/summary.md",
        })

    def test_docs_only_tree_yields_nothing(self):
        import tempfile
        d = tempfile.mkdtemp()
        for name in ("README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE"):
            with open(os.path.join(d, name), "w") as fh:
                fh.write("# docs\nplain prose, no template vars")
        self.assertEqual(discover([d]), [])

    def test_explicit_file_always_honored(self):
        # A file named directly is returned even if a scan would skip it.
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "README.md")
        with open(p, "w") as fh:
            fh.write("docs")
        self.assertEqual(discover([p]), [p])


class TestAutoGPTAdapter(unittest.TestCase):
    """AutoGPT graph exports must be parsed faithfully, not silently mis-read."""

    AG = os.path.join(ROOT, "examples", "autogpt_agent.json")

    def test_detected_as_autogpt(self):
        wf = parse_file(self.AG)
        self.assertEqual(wf.source_kind, "autogpt")

    def test_block_kinds_and_models(self):
        wf = parse_file(self.AG)
        by_id = {n.node_id[:8]: n for n in wf.nodes}
        ai1 = by_id["11111111"]
        tool = by_id["22222222"]
        ai2 = by_id["33333333"]
        # LLM blocks -> llm_call with the REAL model (normalized to catalog ids).
        self.assertEqual(ai1.kind, "llm_call")
        self.assertEqual(ai1.intended_model, "gpt-4o")
        self.assertEqual(ai2.kind, "llm_call")
        self.assertEqual(ai2.intended_model, "claude-sonnet-4")
        # A non-LLM block is a tool, with no fabricated model.
        self.assertEqual(tool.kind, "tool")
        self.assertIsNone(tool.intended_model)

    def test_links_become_edges_and_cycle_detected(self):
        wf = parse_file(self.AG)
        self.assertEqual(len(wf.edges), 3)
        self.assertTrue(len(find_cycles(wf)) >= 1)

    def test_cycle_blocks_and_recs_only_on_llm_nodes(self):
        wf = parse_file(self.AG)
        res = analyze_workflow(wf, cfg=Config(trials=400))
        self.assertIn("recursive_loop", {f.category for f in res.findings})
        self.assertEqual(res.risk.gate_decision, "block")
        # No model-swap rec should target the non-LLM tool node.
        rec_nodes = {r.node_id for r in res.recommendations}
        self.assertNotIn("22222222-2222-2222-2222-222222222222", rec_nodes)

    def test_model_normalization(self):
        from tollgate.parsers.autogpt import _normalize_model
        cases = {
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4-turbo": "gpt-4o",
            "o1-mini": "gpt-4.1-mini",
            "claude-3-haiku-20240307": "claude-haiku-3.5",
            "claude-3-opus-20240229": "claude-opus-4",
            "gemini-1.5-flash": "gemini-1.5-flash",
            "llama-3.1-8b-instant": "llama-3.1-8b",
            "mixtral-8x7b-32768": "mixtral-8x7b",
            # newly-mapped families
            "gpt-5-2025-08-07": "gpt-5",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5-nano": "gpt-5-nano",
            "perplexity/sonar-pro": "sonar-pro",
            "perplexity/sonar-deep-research": "sonar-deep-research",
            "sonar": "sonar",
        }
        for raw, expected in cases.items():
            self.assertEqual(_normalize_model(raw), expected, raw)

    def test_new_models_in_catalog(self):
        cat = ModelCatalog.load()
        for mid in ("gpt-5", "gpt-5-mini", "gpt-5-nano",
                    "sonar", "sonar-pro", "sonar-deep-research"):
            self.assertIsNotNone(cat.get(mid), mid)
        # gpt-5 must have a cheaper safe alternative for substitution to fire.
        self.assertTrue(len(cat.cheaper_alternatives("gpt-5", 0.7)) >= 1)

    def test_image_block_is_tool_not_llm(self):
        import json as _json, tempfile
        d = tempfile.mkdtemp()
        graph = {
            "id": "img", "name": "image_agent",
            "nodes": [
                {"id": "p", "block_id": "AITextGeneratorBlock",
                 "input_default": {"model": "gpt-4o", "prompt": "Describe a scene."}},
                {"id": "img", "block_id": "AIImageGeneratorBlock",
                 "input_default": {"model": "Flux 1.1 Pro Ultra", "prompt": "a cat"}},
            ],
            "links": [{"source_id": "p", "sink_id": "img",
                       "source_name": "out", "sink_name": "prompt"}],
        }
        p = os.path.join(d, "g.json")
        with open(p, "w") as fh:
            _json.dump(graph, fh)
        wf = parse_file(p)
        kinds = {n.node_id: n.kind for n in wf.nodes}
        self.assertEqual(kinds["p"], "llm_call")
        self.assertEqual(kinds["img"], "tool")   # Flux image block, not an LLM
        # The image node must not get a model-swap recommendation.
        res = analyze_workflow(wf, cfg=Config(trials=200))
        self.assertNotIn("img", {r.node_id for r in res.recommendations})


class TestImperativeAdapter(unittest.TestCase):
    """Hand-rolled loop agents (BabyAGI-class) must be discovered and gated.

    These agents are a plain ``while True`` around LLM SDK calls — no framework
    graph. They are the canonical unbounded-cost risk, so an empty result here
    (the old behavior) would read as a false green light.
    """

    BABY = os.path.join(ROOT, "examples", "babyagi_agent.py")

    def test_detected_as_imperative(self):
        wf = parse_file(self.BABY)
        self.assertEqual(wf.source_kind, "imperative")

    def test_llm_calls_recovered_through_wrapper(self):
        # The agents call an ``openai_call`` wrapper, not the SDK directly.
        # Transitive detection must still recover them as llm_call nodes.
        wf = parse_file(self.BABY)
        llm = [n for n in wf.nodes if n.kind == "llm_call"]
        self.assertGreaterEqual(len(llm), 2)
        # Models flow through the wrapper's model= kwarg.
        models = {n.intended_model for n in llm}
        self.assertTrue(any(m and m.startswith("gpt-4o") for m in models), models)

    def test_unbounded_loop_blocks(self):
        wf = parse_file(self.BABY)
        self.assertTrue(len(find_cycles(wf)) >= 1)
        res = analyze_workflow(wf, cfg=Config(trials=400))
        cats = {f.category for f in res.findings}
        self.assertIn("recursive_loop", cats)
        loop = [f for f in res.findings if f.category == "recursive_loop"]
        self.assertTrue(any(f.severity == "critical" for f in loop))
        self.assertEqual(res.risk.gate_decision, "block")

    def test_discovered_in_directory_scan(self):
        found = {os.path.basename(p) for p in discover([os.path.dirname(self.BABY)])}
        self.assertIn("babyagi_agent.py", found)

    def test_has_imperative_llm_signal(self):
        from tollgate.parsers import has_imperative_llm
        self.assertTrue(has_imperative_llm("openai.chat.completions.create(x)"))
        self.assertFalse(has_imperative_llm("print('hello')"))

    def test_bounded_loop_is_not_critical(self):
        # while True with a break is a bounded self-loop: flagged, not critical.
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "guarded.py")
        with open(p, "w") as fh:
            fh.write(
                "import openai\n"
                "def step(x):\n"
                "    return openai.chat.completions.create(model='gpt-4o',\n"
                "        messages=[{'role':'user','content':x}])\n"
                "def main():\n"
                "    i = 0\n"
                "    while True:\n"
                "        step('go')\n"
                "        i += 1\n"
                "        if i > 5:\n"
                "            break\n"
            )
        wf = parse_file(p)
        self.assertEqual(wf.source_kind, "imperative")
        res = analyze_workflow(wf, cfg=Config(trials=200))
        loop = [f for f in res.findings if f.category == "recursive_loop"]
        # A bound exists, so no critical loop finding.
        self.assertFalse(any(f.severity == "critical" for f in loop))

    def test_sequential_script_has_no_cycle(self):
        # A straight-line script (no loop) must not fabricate a recursive loop.
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "seq.py")
        with open(p, "w") as fh:
            fh.write(
                "import openai\n"
                "def ask(x):\n"
                "    return openai.chat.completions.create(model='gpt-4o',\n"
                "        messages=[{'role':'user','content':x}])\n"
                "def main():\n"
                "    a = ask('one')\n"
                "    b = ask('two')\n"
                "    print(a, b)\n"
            )
        wf = parse_file(p)
        self.assertEqual(wf.source_kind, "imperative")
        self.assertEqual(find_cycles(wf), [])


class TestBroadSDKCoverage(unittest.TestCase):
    """Beyond OpenAI: a hand-rolled loop around any recognized provider SDK must
    be discovered as imperative and have its cycle recovered. OpenAI-compatible
    vendors (Groq, Together, DeepSeek, ...) reuse the OpenAI shape and are covered
    by that marker; the non-OpenAI-shaped SDKs are checked explicitly here."""

    # label -> a single SDK call expression as it appears inside the loop body.
    CALLS = {
        "openai_compat_groq": "client.chat.completions.create(model='llama-3.1-8b', messages=m)",
        "openai_responses": "client.responses.create(model='gpt-4.1', input=m)",
        "anthropic": "client.messages.create(model='claude-sonnet-4', messages=m, max_tokens=10)",
        "gemini": "gmodel.generate_content(prompt)",
        "gemini_async": "gmodel.generate_content_async(prompt)",
        "mistral_native": "client.chat.complete(model='mistral-large-latest', messages=m)",
        "mistral_stream": "client.chat.stream(model='mistral-large-latest', messages=m)",
        "bedrock_converse": "brt.converse(modelId='anthropic.claude-3', messages=m)",
        "bedrock_invoke": "brt.invoke_model(modelId='anthropic.claude-3', body=b)",
        "cohere_stream": "co.chat_stream(model='command-r', message=x)",
        "replicate": "replicate.run('meta/meta-llama-3-8b', input={'prompt': x})",
        "ollama": "ollama.chat(model='llama3', messages=m)",
        "ollama_generate": "ollama.generate(model='llama3', prompt=x)",
        "hf_chat": "hf.chat_completion(messages=m, model='meta-llama/Llama-3')",
        "hf_textgen": "hf.text_generation(x, model='bigscience/bloom')",
        "litellm": "litellm.completion(model='gpt-4o', messages=m)",
    }

    def _write_loop_agent(self, call_expr):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "agent.py")
        with open(p, "w") as fh:
            fh.write(
                "m = [{'role': 'user', 'content': 'go'}]\n"
                "b = '{}'\n"
                "prompt = x = 'go'\n"
                "def main():\n"
                "    while True:\n"
                "        r = " + call_expr + "\n"
                "        m.append(r)\n"
            )
        return p

    def test_each_sdk_shape_is_discovered_and_looped(self):
        for label, expr in self.CALLS.items():
            with self.subTest(sdk=label):
                p = self._write_loop_agent(expr)
                # Discovered as an imperative agent (not dropped, not a prompt).
                self.assertIn(p, discover([os.path.dirname(p)]),
                              f"{label}: file not discovered as a workflow candidate")
                wf = parse_file(p)
                self.assertEqual(wf.source_kind, "imperative",
                                 f"{label}: not routed to the imperative parser")
                # The unbounded `while True` cycle around the call is recovered.
                self.assertTrue(find_cycles(wf),
                                f"{label}: loop-around-SDK structure not recovered")

    def test_has_imperative_llm_signal_new_shapes(self):
        from tollgate.parsers import has_imperative_llm
        for expr in self.CALLS.values():
            self.assertTrue(has_imperative_llm(expr), expr)

    def test_bedrock_camelcase_model_id_is_recovered(self):
        # Bedrock passes the model as modelId=; costing/substitution need it bound.
        p = self._write_loop_agent(
            "brt.converse(modelId='anthropic.claude-3', messages=m)")
        wf = parse_file(p)
        models = {n.intended_model for n in wf.nodes if n.intended_model}
        self.assertTrue(models, "modelId= was not picked up as the node's model")


class TestHonestFailure(unittest.TestCase):
    """An unrecognized nodes-only JSON must not masquerade as a confident PASS."""

    def test_structureless_json_is_dropped(self):
        import json as _json
        import tempfile
        d = tempfile.mkdtemp()
        # Foreign schema: has 'nodes' (so discovery picks it up) but no LLM
        # signal and no resolvable edges -> not analyzable.
        junk = {"nodes": [{"id": "a", "block_id": "HttpRequestBlock",
                           "input_default": {"url": "x"}}]}
        with open(os.path.join(d, "thing.json"), "w") as fh:
            _json.dump(junk, fh)
        run = analyze_path([d], cfg=Config(trials=200))
        self.assertEqual(len(run.results), 0)
        self.assertEqual(run.gate_decision, "pass")  # empty run, not a false PASS row


class TestCrewAIParser(unittest.TestCase):
    """CrewAI files have no LangGraph builder and no raw SDK call; before the
    dedicated parser they were silently dropped (workflow_count=0). These lock in
    that they are now discovered and that delegation loops are flagged."""

    def _write(self, body):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "crew.py")
        with open(p, "w") as fh:
            fh.write(body)
        return d, p

    HIER = (
        "from crewai import Agent, Task, Crew, Process\n"
        "researcher = Agent(role='R', goal='g', backstory='b', llm='gpt-4o', allow_delegation=True)\n"
        "writer = Agent(role='W', goal='g', backstory='b', llm='gpt-4o-mini')\n"
        "t1 = Task(description='Research the landscape thoroughly.', agent=researcher)\n"
        "t2 = Task(description='Write a long report.', agent=writer)\n"
        "crew = Crew(agents=[researcher, writer], tasks=[t1, t2],\n"
        "            process=Process.hierarchical, memory=True, manager_llm='gpt-4o')\n"
        "crew.kickoff()\n"
    )
    SEQ = (
        "from crewai import Agent, Task, Crew\n"
        "a = Agent(role='C', goal='g', backstory='b', llm='gpt-4o-mini')\n"
        "t = Task(description='Classify the ticket.', agent=a)\n"
        "crew = Crew(agents=[a], tasks=[t])\n"
        "crew.kickoff()\n"
    )
    DECORATOR = (
        "from crewai import Agent, Task, Crew, Process\n"
        "from crewai.project import CrewBase, agent, task, crew\n"
        "@CrewBase\n"
        "class MyCrew:\n"
        "    @agent\n"
        "    def researcher(self):\n"
        "        return Agent(role='R', goal='g', backstory='b', llm='gpt-4o', allow_delegation=True)\n"
        "    @task\n"
        "    def research(self):\n"
        "        return Task(description='Do deep research now.', agent=self.researcher())\n"
        "    @crew\n"
        "    def crew(self):\n"
        "        return Crew(agents=self.agents, tasks=self.tasks, process=Process.hierarchical)\n"
    )

    def test_detected_as_crewai(self):
        _d, p = self._write(self.SEQ)
        wf = parse_file(p)
        self.assertEqual(wf.source_kind, "crewai")
        self.assertTrue(wf.llm_nodes())

    def test_discovery_picks_up_crewai_file(self):
        d, _p = self._write(self.HIER)
        found = discover([d])
        self.assertEqual(len(found), 1)

    def test_hierarchical_crew_flags_unbounded_delegation_loop(self):
        d, _p = self._write(self.HIER)
        run = analyze_path([d], cfg=Config(trials=300))
        self.assertEqual(len(run.results), 1)
        cats = [f["category"] for r in run.results for f in r.to_dict()["findings"]]
        self.assertIn("recursive_loop", cats)
        self.assertEqual(run.gate_decision, "block")
        # The recovered model is read from the agent's llm=, not guessed.
        wf = parse_file(os.path.join(d, "crew.py"))
        self.assertIn("gpt-4o", {n.intended_model for n in wf.nodes})

    def test_sequential_crew_has_no_loop(self):
        d, _p = self._write(self.SEQ)
        run = analyze_path([d], cfg=Config(trials=300))
        self.assertEqual(len(run.results), 1)
        cats = [f["category"] for r in run.results for f in r.to_dict()["findings"]]
        self.assertNotIn("recursive_loop", cats)

    def test_decorator_style_is_recovered_and_looped(self):
        d, _p = self._write(self.DECORATOR)
        wf = parse_file(os.path.join(d, "crew.py"))
        self.assertTrue(wf.nodes, "decorator-style crew should yield >=1 task node")
        self.assertTrue(find_cycles(wf), "delegation crew should expose a cycle")

    def test_empty_crew_is_honestly_dropped(self):
        # crewai import + Crew() but nothing to recover -> not a false PASS row.
        d, _p = self._write("from crewai import Crew\ncrew = Crew()\ncrew.kickoff()\n")
        run = analyze_path([d], cfg=Config(trials=100))
        self.assertEqual(len(run.results), 0)


class TestAgenticLint(unittest.TestCase):
    """Strict source-level agentic linter: reaches frameworks we don't graph,
    flags config-absence risks, and stays silent on non-agentic code."""

    def _run(self, body, strictness="strict"):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "agent.py")
        with open(p, "w") as fh:
            fh.write(body)
        return analyze_path([d], cfg=Config(trials=150, lint_strictness=strictness))

    def _lint_cats(self, run):
        return {f["category"] for lr in run.to_dict()["lint_results"] for f in lr["findings"]}

    def test_langchain_agentexecutor_missing_cap(self):
        body = ("from langchain.agents import AgentExecutor\n"
                "ex = AgentExecutor(agent=a, tools=t)\n"
                "ex.invoke({'input': 'x'})\n")
        run = self._run(body)
        self.assertIn("missing_iteration_cap", self._lint_cats(run))
        self.assertEqual(run.gate_decision, "warn")

    def test_langchain_cap_present_is_clean(self):
        body = ("from langchain.agents import AgentExecutor\n"
                "ex = AgentExecutor(agent=a, tools=t, max_iterations=6)\n"
                "ex.invoke({'input': 'x'})\n")
        run = self._run(body)
        self.assertNotIn("missing_iteration_cap", self._lint_cats(run))

    def test_autogen_groupchat_missing_round(self):
        body = ("import autogen\n"
                "gc = autogen.GroupChat(agents=[x, y], messages=[])\n")
        run = self._run(body)
        self.assertIn("missing_iteration_cap", self._lint_cats(run))

    def test_llamaindex_react_missing_iterations(self):
        body = ("from llama_index.core.agent import ReActAgent\n"
                "agent = ReActAgent(tools=t, llm=llm)\n")
        run = self._run(body)
        self.assertIn("missing_iteration_cap", self._lint_cats(run))

    def test_smolagents_missing_max_steps(self):
        body = ("from smolagents import CodeAgent\n"
                "a = CodeAgent(tools=t, model=m)\n")
        run = self._run(body)
        self.assertIn("missing_iteration_cap", self._lint_cats(run))

    def test_langgraph_missing_recursion_limit(self):
        body = ("from langgraph.graph import StateGraph\n"
                "g = StateGraph(dict)\n"
                "g.add_node('a', f)\n"
                "app = g.compile()\n"
                "app.invoke({'x': 1})\n")
        # LangGraph parses as a graph; the lint recursion_limit check merges in.
        run = analyze_path([self._tmp(body)], cfg=Config(trials=150))
        cats = {f["category"] for r in run.to_dict()["results"] for f in r["findings"]}
        self.assertIn("missing_iteration_cap", cats)

    def test_uncapped_output_flagged_on_raw_call(self):
        body = ("from openai import OpenAI\n"
                "c = OpenAI()\n"
                "def f():\n"
                "    return c.chat.completions.create(model='gpt-4o', messages=m)\n")
        run = analyze_path([self._tmp(body)], cfg=Config(trials=150))
        cats = {f["category"] for r in run.to_dict()["results"] for f in r["findings"]}
        self.assertIn("uncapped_output", cats)

    def test_capped_output_is_clean(self):
        body = ("from openai import OpenAI\n"
                "c = OpenAI()\n"
                "def f():\n"
                "    return c.chat.completions.create(model='gpt-4o', messages=m, max_tokens=256)\n")
        run = analyze_path([self._tmp(body)], cfg=Config(trials=150))
        cats = {f["category"] for r in run.to_dict()["results"] for f in r["findings"]}
        self.assertNotIn("uncapped_output", cats)

    def test_unbounded_gather_fanout(self):
        body = ("import asyncio\n"
                "from openai import AsyncOpenAI\n"
                "c = AsyncOpenAI()\n"
                "async def one(x):\n"
                "    return await c.chat.completions.create(model='gpt-4o', messages=x, max_tokens=50)\n"
                "async def main(items):\n"
                "    return await asyncio.gather(*[one(x) for x in items])\n")
        run = analyze_path([self._tmp(body)], cfg=Config(trials=150))
        cats = {f["category"] for r in run.to_dict()["results"] for f in r["findings"]}
        self.assertIn("fanout", cats)

    def test_non_agentic_code_is_silent(self):
        body = ("import os\n"
                "def add(a, b):\n"
                "    while True:\n"
                "        if a > b:\n"
                "            break\n"
                "    return a + b\n")
        run = self._run(body)
        d = run.to_dict()
        self.assertEqual(d["workflow_count"], 0)
        self.assertEqual(d["lint_count"], 0)
        self.assertEqual(d["gate_decision"], "pass")

    def test_langchain_chatmodel_no_max_tokens(self):
        # gpt-engineer pattern: model used via .invoke(), cap belongs on the ctor.
        body = ("from langchain_openai import ChatOpenAI\n"
                "llm = ChatOpenAI(model='gpt-4o', temperature=0.1)\n"
                "out = llm.invoke(messages)\n")
        run = self._run(body)
        self.assertIn("uncapped_output", self._lint_cats(run))

    def test_langchain_chatmodel_with_max_tokens_clean(self):
        body = ("from langchain_openai import ChatOpenAI\n"
                "llm = ChatOpenAI(model='gpt-4o', max_tokens=512)\n"
                "out = llm.invoke(messages)\n")
        run = self._run(body)
        self.assertNotIn("uncapped_output", self._lint_cats(run))

    def test_llamaindex_openailike_no_max_tokens(self):
        # RepoAgent pattern: model wrapped in a LlamaIndex LLM class.
        body = ("from llama_index.llms.openai_like import OpenAILike\n"
                "llm = OpenAILike(model='x', api_base='b', api_key='k')\n"
                "out = llm.complete(prompt)\n")
        run = self._run(body)
        self.assertIn("uncapped_output", self._lint_cats(run))

    def test_raw_openai_client_not_flagged_as_model_ctor(self):
        # Collision safety: `from openai import OpenAI` is a client, not a
        # LlamaIndex LLM, and a capped create() must produce no uncapped_output.
        body = ("from openai import OpenAI\n"
                "client = OpenAI()\n"
                "def f():\n"
                "    return client.chat.completions.create(model='gpt-4o', messages=m, max_tokens=100)\n")
        run = analyze_path([self._tmp(body)], cfg=Config(trials=120))
        cats = {f["category"] for r in run.to_dict()["results"] for f in r["findings"]}
        self.assertNotIn("uncapped_output", cats)

    def test_modern_autogen_team_missing_max_turns(self):
        body = ("from autogen_agentchat.teams import RoundRobinGroupChat\n"
                "from autogen_agentchat.agents import AssistantAgent\n"
                "a = AssistantAgent('a')\n"
                "team = RoundRobinGroupChat([a])\n")
        run = self._run(body)
        self.assertIn("missing_iteration_cap", self._lint_cats(run))

    def test_strictness_off_disables_lint(self):
        body = ("from langchain.agents import AgentExecutor\n"
                "ex = AgentExecutor(agent=a, tools=t)\n")
        run = self._run(body, strictness="off")
        self.assertEqual(run.to_dict()["lint_count"], 0)

    def _tmp(self, body):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "agent.py")
        with open(p, "w") as fh:
            fh.write(body)
        return d


class TestFingerprintAndVerify(unittest.TestCase):
    """Self-healing outputs: a tamper-evident fingerprint binds verdict to inputs,
    and `tollgate verify` re-derives it to catch edited/drifted reports."""

    def _wf(self):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "wf.yaml")
        with open(p, "w") as fh:
            fh.write("workflow: w\nentry: a\nnodes:\n  - {id: a, kind: llm_call, model: gpt-4o}\nedges: []\n")
        return d, p

    def test_fingerprint_present_and_deterministic(self):
        _d, p = self._wf()
        a = analyze_path([p], cfg=Config(trials=200))
        b = analyze_path([p], cfg=Config(trials=200))
        self.assertTrue(a.fingerprint)
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertEqual(a.to_dict()["fingerprint"], a.fingerprint)

    def test_verify_clean_then_tampered(self):
        import json
        from tollgate.cli import main
        d, p = self._wf()
        run = analyze_path([p], cfg=Config(trials=200))
        rep = os.path.join(d, "report.json")
        with open(rep, "w") as fh:
            json.dump(run.to_dict(), fh)
        self.assertEqual(main(["verify", rep, p]), 0)
        # Tamper the gate + fingerprint -> verify must fail (non-zero).
        j = json.load(open(rep))
        j["gate_decision"] = "pass"
        j["fingerprint"] = "00" * 32
        with open(rep, "w") as fh:
            json.dump(j, fh)
        self.assertEqual(main(["verify", rep, p]), 1)


class TestTrafficScenarios(unittest.TestCase):
    def test_default_is_single_10k_per_week_scenario(self):
        from tollgate.simulation import DEFAULT_SCENARIOS, scenario_from_volume
        self.assertEqual(len(DEFAULT_SCENARIOS), 1)  # no peak_2x / viral_10x
        base = DEFAULT_SCENARIOS[0]
        self.assertAlmostEqual(base.rps, 10000 / (7 * 86400.0), places=6)
        self.assertEqual(base.diurnal_peak_multiplier, 1.0)

    def test_scenario_from_volume_day_vs_week(self):
        from tollgate.simulation import scenario_from_volume
        wk = scenario_from_volume(10000, "week")
        dy = scenario_from_volume(10000, "day")
        self.assertAlmostEqual(dy.rps, 10000 / 86400.0, places=6)
        self.assertAlmostEqual(wk.rps, 10000 / (7 * 86400.0), places=6)
        self.assertGreater(dy.rps, wk.rps)

    def test_config_requests_per_week_and_day_keys(self):
        from tollgate.pipeline import _scenarios_from_config, _rps_from_scenario
        wk = _rps_from_scenario({"requests_per_week": 10000})
        dy = _rps_from_scenario({"requests_per_day": 1500})
        self.assertAlmostEqual(wk, 10000 / (7 * 86400.0), places=6)
        self.assertAlmostEqual(dy, 1500 / 86400.0, places=6)
        # explicit rps still wins (back-compat)
        self.assertEqual(_rps_from_scenario({"rps": 2.5, "requests_per_week": 1}), 2.5)
        # flows through the config -> scenario path
        cfg = Config(scenarios=[{"name": "s", "requests_per_week": 70000, "horizon_days": 7}])
        sc = _scenarios_from_config(cfg)
        self.assertAlmostEqual(sc[0].rps, 70000 / (7 * 86400.0), places=6)

    def test_cli_traffic_override(self):
        from tollgate.cli import main
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "wf.yaml")
        with open(p, "w") as fh:
            fh.write("workflow: w\nentry: a\nnodes:\n  - {id: a, kind: llm_call, model: gpt-4o}\n")
        # Should run without error and honor the per-week override.
        rc = main(["analyze", p, "--traffic-per-week", "70000", "-f", "json"])
        self.assertIn(rc, (0, 1))  # pass or block, but a clean run


class TestPromptScan(unittest.TestCase):
    """Language-agnostic prompt mining: find prompts hidden in source/config of
    any language, and stay quiet on non-prompt strings."""

    def _scan(self, name, src):
        from tollgate.prompt_scan import scan_text
        return scan_text(name, src)

    def test_detects_python_constant(self):
        r = self._scan("prompts.py",
            'SYSTEM_PROMPT = """You are a senior QA engineer. Your task is to read the '
            'PRD and produce test cases. Respond only with JSON. Do not add commentary."""')
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].name, "SYSTEM_PROMPT")

    def test_detects_js_template_literal(self):
        r = self._scan("p.ts", 'const systemPrompt = `You are a helpful assistant. '
                               'Answer the user step by step and format output as JSON.`;')
        self.assertTrue(r and r[0].name == "systemPrompt")

    def test_detects_go_raw_string(self):
        r = self._scan("p.go", 'var planPrompt = `You are an expert planner. As an AI '
                               'agent, analyze the requirements and respond with a plan.`')
        self.assertTrue(r)

    def test_detects_yaml_value(self):
        r = self._scan("c.yaml", 'system_prompt: "You are Aegis, an autonomous testing '
                                 'agent. You must analyze the repo and do not fabricate results."')
        self.assertTrue(r)

    def test_detects_ruby_heredoc(self):
        r = self._scan("a.rb", 'PROMPT = <<~SYS\n  You are an assistant. Your job is to '
                               'summarize the diff and reply with a short review.\nSYS')
        self.assertTrue(r)

    def test_ignores_sql(self):
        r = self._scan("q.py", 'SQL = "SELECT id, name, email FROM users WHERE active = 1 '
                               'AND created_at > now() ORDER BY name LIMIT 100"')
        self.assertEqual(r, [])

    def test_ignores_html(self):
        r = self._scan("v.js", 'const h = "<div class=\\"card\\"><span>Hi</span></div>'
                               '<p>more text here</p><a href=\\"x\\">link</a><b>bold</b>"')
        self.assertEqual(r, [])

    def test_ignores_log_format_string(self):
        r = self._scan("l.py", 'm = "processed %s items in %s ms for tenant %s status %s code %s"')
        self.assertEqual(r, [])

    def test_ignores_short_strings(self):
        self.assertEqual(self._scan("x.py", 'NAME = "ok then"'), [])

    def test_pipeline_surfaces_detected_prompts_advisory(self):
        import tempfile
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "prompts.py"), "w") as fh:
            fh.write('SYSTEM_PROMPT = """You are an expert reviewer. Your task is to '
                     'analyze the code and respond only with structured JSON output."""\n')
        run = analyze_path([d], cfg=Config(trials=80))
        dd = run.to_dict()
        self.assertEqual(dd["detected_prompt_count"], 1)
        self.assertEqual(dd["gate_decision"], "pass")  # advisory: never blocks

    def test_detected_prompt_gets_efficiency_review(self):
        # A bloated embedded prompt should be reviewed and surface in the
        # dashboard's prompt-optimisation table, not just the detected list.
        import tempfile
        from tollgate import html_report
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "prompts.py"), "w") as fh:
            fh.write('SYSTEM_PROMPT = """You are a very very helpful assistant. Please be '
                     'sure to always respond. Your task is to absolutely and definitely '
                     'analyze the input in order to produce JSON only. Do not ever add '
                     'commentary. You are a very very helpful assistant."""\n')
        run = analyze_path([d], cfg=Config(trials=60))
        self.assertTrue(run.detected_prompts)
        self.assertIsNotNone(run.detected_prompts[0].review)
        data = html_report.build_dashboard_data(run)
        self.assertTrue(any("embedded" in r["wf"] for r in data["prompt_reviews"]))

    def test_chat_template_machinery_not_reviewed(self):
        # A Jinja chat template is control flow, not a prose prompt — the
        # efficiency rewriter must not touch it (it would corrupt the template).
        from tollgate.prompt_review import review_text
        tmpl = ("{%- for message in messages -%}\n"
                "{%- if message['role'] == 'system' -%}\n"
                "{{ message['content'] }}\n"
                "{%- endif -%}\n"
                "{%- endfor -%}\n"
                "{{'You are an AI programming assistant.'}}")
        self.assertIsNone(review_text(tmpl))
        # but a prose prompt that merely interpolates one variable is still reviewed
        prose = ("You are a very very helpful assistant. Please be sure to always "
                 "respond. Context: {{context}}. Do not ever add commentary.")
        self.assertIsNotNone(review_text(prose))

    def test_prompt_scan_disabled(self):
        import tempfile
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "prompts.py"), "w") as fh:
            fh.write('SYSTEM_PROMPT = """You are an expert. Your task is to respond '
                     'with JSON only and never add commentary to your answers."""\n')
        run = analyze_path([d], cfg=Config(trials=80, prompt_scan=False))
        self.assertEqual(run.to_dict()["detected_prompt_count"], 0)


class TestDashboardScriptOrdering(unittest.TestCase):
    """Guard against use-before-definition in the dashboard's inline JS (a `const`
    helper called above its declaration throws a TDZ error at runtime and blanks
    the whole page — which a Python build can't catch)."""

    def _html_with_prompts(self):
        import tempfile
        from tollgate import html_report
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "prompts.py"), "w") as fh:
            fh.write('SYSTEM_PROMPT = """You are an expert reviewer. Your task is to '
                     'analyze the code and respond only with structured JSON output."""\n')
        with open(os.path.join(d, "agent.py"), "w") as fh:
            fh.write("import openai\n"
                     "def run():\n"
                     "    while True:\n"
                     "        openai.chat.completions.create(model='gpt-4o', messages=m)\n")
        run = analyze_path([d], cfg=Config(trials=80))
        return html_report.to_html(run)

    def test_no_helper_called_before_declaration(self):
        html = self._html_with_prompts()
        # the detected-prompts data + a workflow must both be present
        self.assertIn("detected_prompts", html)
        for helper in ("fmtBig", "esc", "loc"):
            decl = html.find("const " + helper)
            call = html.find(helper + "(")
            if decl != -1 and call != -1:
                self.assertLessEqual(decl, call,
                    f"`{helper}` is called before its declaration in the dashboard JS")


class TestReportFormats(unittest.TestCase):
    def test_all_formats_render(self):
        from tollgate import report
        run = analyze_path([EXAMPLES], cfg=Config(trials=400))
        self.assertIn("Tollgate", report.to_markdown(run))
        self.assertIn("gate_decision", report.to_json(run))
        import json
        json.loads(report.to_sarif(run))
        json.loads(report.to_gitlab_codequality(run))
        self.assertIn("gate", report.to_terminal(run))

    def test_html_dashboard(self):
        import json
        from tollgate import report
        from tollgate.html_report import build_dashboard_data
        run = analyze_path([EXAMPLES], cfg=Config(trials=400))

        html = report.to_html(run)
        # Self-contained, single-file dashboard with the data inlined.
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Chart.js", html)
        self.assertNotIn("__TOLLGATE_DATA__", html)  # placeholder must be substituted

        # The injected blob must be valid JSON and match the report payload.
        blob = html.split("const D = ", 1)[1].split(";\n", 1)[0]
        embedded = json.loads(blob)
        data = build_dashboard_data(run)
        self.assertEqual(embedded["gate"], run.gate_decision)
        self.assertEqual(embedded["gate"], data["gate"])
        self.assertEqual(embedded["max_score"], run.max_score)
        self.assertEqual(embedded["workflows"], len(run.results))
        # Dashboard totals must agree with the run.
        self.assertEqual(
            data["total_findings"], sum(len(r.findings) for r in run.results))
        self.assertEqual(
            data["total_recs"], sum(len(r.recommendations) for r in run.results))
        for row in data["rows"]:
            self.assertIn(row["gate"], ("pass", "warn", "block"))


if __name__ == "__main__":
    unittest.main()
