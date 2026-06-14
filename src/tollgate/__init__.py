"""Tollgate — prevention-first token-risk analysis for AI agents, for CI/CD.

Public API:
    from tollgate import analyze_path, analyze_workflow
    from tollgate.catalog import ModelCatalog
"""
from .version import __version__
from .pipeline import analyze_path, analyze_workflow, AnalysisResult

__all__ = ["__version__", "analyze_path", "analyze_workflow", "AnalysisResult"]
