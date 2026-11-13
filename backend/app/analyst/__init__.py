"""Analyst Copilot: a model-orchestrated analytical assistant over one dataset.

Separate from the recursive Planner -> Executor -> Critic loop (app/agents) and
from the rules-only Copilot session reader (app/copilot.py); neither is used or
changed here.

The flow for one message (see orchestrator.ask):

    1. plan      the model (a fine-tuned local model when served, otherwise the
                 deterministic planner in planner.py) returns a JSON plan:
                 a conversational reply, a clarification, or tool calls.
    2. validate  every tool call is checked against the tool's schema and the
                 dataset's real columns (tools.validate_call). Invalid calls are
                 never executed.
    3. execute   tools run deterministically (metric engine, statistics,
                 scikit-learn Random Forest, scenario arithmetic).
    4. explain   the answer is written from the tool results. A model-written
                 narrative is accepted only if every number in it occurs in the
                 tool results (explain.grounded); otherwise the deterministic
                 rendering is used.

No number in an answer comes from a model: every figure is a tool output.
"""
