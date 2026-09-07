"""Structured-ish logging setup, called once at app startup.

Replaces ad-hoc print() calls scattered through the agent pipeline with
real logger calls that carry a level and a module name, and go through one
place if this ever needs to point at a log aggregator instead of stdout.
"""
from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    # Quiet third-party libraries down to warnings unless the operator asked
    # for verbose output explicitly -- otherwise INFO is dominated by
    # per-request access logs and HTTP client chatter.
    if level != "DEBUG":
        for noisy in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
