"""Comprehensive cross-language prompt-detection corpus.

30 realistic ways an LLM prompt hides in source, across the most common languages
and string forms (triple-quotes, template literals, heredocs/nowdocs, text
blocks, raw/verbatim/interpolated strings, YAML/JSON config, shell). Every
POSITIVE must be detected; a batch of NEGATIVES (SQL, HTML, logs, URLs, regex,
base64, code) must NOT be flagged.

Run: python -m unittest tests.test_prompt_scan_languages
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tollgate.prompt_scan import scan_text  # noqa: E402

# (id, filename, source) — each contains exactly one embedded prompt to find.
POSITIVES = [
    # --- Python ---
    ("py_triple_double", "prompts.py",
     'SYSTEM_PROMPT = """You are a senior QA engineer. Your task is to read the PRD '
     'and produce thorough test cases. Respond only with JSON, no commentary."""'),
    ("py_triple_single", "prompts.py",
     "PLAN = '''You are an expert planner. Analyze the request step by step and "
     "respond with a numbered plan. Do not include extra prose.'''"),
    ("py_single_double", "prompts.py",
     'SYSTEM = "You are a helpful assistant. Answer the user concisely and never fabricate facts or sources."'),
    ("py_fstring_triple", "prompts.py",
     'prompt = f"""You are {role}. Your job is to summarize the diff and reply with a short structured review."""'),
    ("py_messages_dict", "agent.py",
     'msgs = [{"role": "system", "content": "You are Aegis, an autonomous testing agent. '
     'You must analyze the repository and do not invent results."}]'),

    # --- JavaScript / TypeScript ---
    ("js_backtick", "prompts.ts",
     'const systemPrompt = `You are a helpful coding assistant. Answer the user step '
     'by step and always format your final output as JSON.`;'),
    ("ts_typed_const", "prompts.ts",
     'const prompt: string = "You are an expert reviewer. Respond with concise, actionable feedback only.";'),
    ("js_double_long", "p.js",
     'export const SYS = "You are a translation assistant. Translate the user message and do not add commentary.";'),
    ("ts_template_var", "p.tsx",
     'const p = `You are an assistant. Use the following context: ${ctx}. Answer the question truthfully.`;'),

    # --- Go ---
    ("go_raw_string", "prompts.go",
     'var planPrompt = `You are an expert test planner. As an AI agent, analyze the '
     'requirements and respond with a numbered plan.`'),
    ("go_const_double", "prompts.go",
     'const systemPrompt = "You are a Go expert. Answer the question and provide a minimal code example only."'),

    # --- Java / Kotlin ---
    ("java_text_block", "Prompts.java",
     'String SYSTEM_PROMPT = """\nYou are a senior engineer. Review the code and '
     'respond with a JSON list of issues. Do not add prose.\n""";'),
    ("java_string", "Prompts.java",
     'static final String PROMPT = "You are an assistant. Summarize the document and respond in under 100 words.";'),
    ("kotlin_triple", "Prompts.kt",
     'val systemPrompt = """You are a helpful assistant. Your task is to classify the '
     'ticket and respond with one category word only."""'),
    ("kotlin_val", "Prompts.kt",
     'val prompt = "You are an expert. Answer the user question and cite the relevant section of the context."'),

    # --- Ruby / PHP / shell heredocs ---
    ("ruby_heredoc", "agent.rb",
     "PROMPT = <<~SYS\n  You are an assistant. Your job is to summarize the diff and "
     "reply with a short review. Do not add commentary.\nSYS"),
    ("ruby_single", "agent.rb",
     'PROMPT = "You are a Ruby expert. Respond with a corrected snippet and a one line explanation only."'),
    ("php_heredoc", "prompts.php",
     "$prompt = <<<SYS\nYou are an assistant. Analyze the request and respond with "
     "valid JSON only. Do not include any prose.\nSYS;"),
    ("php_double", "prompts.php",
     '$system = "You are a PHP expert. Answer the user and return only the corrected code, no commentary.";'),
    ("bash_heredoc", "run.sh",
     'read -r -d "" PROMPT <<EOF\nYou are a shell assistant. Explain the command and '
     'respond with a short, safe one-liner.\nEOF'),
    ("bash_assign", "run.sh",
     'PROMPT="You are an assistant. Summarize the build log and respond with the top three errors only."'),

    # --- C# / Swift / Rust ---
    ("csharp_verbatim", "Prompts.cs",
     'string prompt = @"You are an assistant. Read the issue and respond with a step by step fix plan.";'),
    ("csharp_interp", "Prompts.cs",
     'var system = $"You are an expert. Use the context {ctx} and answer the user question truthfully.";'),
    ("swift_multiline", "Prompts.swift",
     'let systemPrompt = """\nYou are a helpful assistant. Answer the user and respond '
     'with a concise summary only.\n"""'),
    ("swift_let", "Prompts.swift",
     'let prompt = "You are a Swift expert. Provide a minimal example and a one sentence explanation only."'),
    ("rust_raw", "prompts.rs",
     'let prompt = r#"You are an assistant. Analyze the code and respond with a JSON array of suggestions."#;'),
    ("rust_plain", "prompts.rs",
     'const SYSTEM: &str = "You are a Rust expert. Answer the question and return only safe, idiomatic code.";'),

    # --- config formats ---
    ("yaml_inline", "config.yaml",
     'system_prompt: "You are Aegis, an autonomous agent. You must analyze the repo and do not fabricate results."'),
    ("yaml_block_scalar", "config.yaml",
     "system_prompt: |\n  You are a helpful assistant. Your task is to read the input\n"
     "  and respond with a structured JSON answer. Do not add commentary.\n"),
    ("json_value", "config.json",
     '{ "system_prompt": "You are an expert assistant. Answer the user question and respond with JSON only." }'),
]

NEGATIVES = [
    ("sql", "q.py", 'SQL = "SELECT id, name, email FROM users WHERE active = 1 AND '
                    'created_at > now() ORDER BY name LIMIT 100"'),
    ("html", "v.js", 'const h = "<div class=\\"card\\"><span>Hello</span></div>'
                     '<p>more text</p><a href=\\"x\\">link</a><b>bold</b>"'),
    ("log_fmt", "l.py", 'm = "processed %s items in %s ms for tenant %s status %s code %s done"'),
    ("url", "u.py", 'U = "https://example.com/very/long/path?a=1&b=2&c=3&d=4&e=5&f=6&g=7&h=8&i=9"'),
    ("regex", "r.py", 'PAT = "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{2,}$"'),
    ("code", "c.js", 'const s = "function add(a,b){return a+b;} const x = add(1,2); console.log(x);"'),
]


class TestCrossLanguagePromptDetection(unittest.TestCase):
    def test_all_positives_detected(self):
        missed = []
        for cid, fn, src in POSITIVES:
            if not scan_text(fn, src):
                missed.append(cid)
        self.assertEqual(missed, [], f"failed to detect prompts in: {missed}")

    def test_positive_count(self):
        self.assertGreaterEqual(len(POSITIVES), 30)

    def test_negatives_not_flagged(self):
        flagged = []
        for cid, fn, src in NEGATIVES:
            if scan_text(fn, src):
                flagged.append(cid)
        self.assertEqual(flagged, [], f"false positives on: {flagged}")

    def test_each_positive_individually(self):
        # Sub-test per case so a failure names the exact language/form.
        for cid, fn, src in POSITIVES:
            with self.subTest(case=cid):
                self.assertTrue(scan_text(fn, src), f"{cid}: not detected")


if __name__ == "__main__":
    unittest.main()
