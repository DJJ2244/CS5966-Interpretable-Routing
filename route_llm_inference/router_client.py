"""
router_client.py - Shared RouteLLM controller instance.

Import `client`, `ROUTER`, and `THRESHOLD` from here to guarantee that
toughness scores are computed with exactly the same router as inference.
"""

import os
from routellm.controller import Controller

WEAK_MODEL   = os.environ["WEAK_MODEL"]
STRONG_MODEL = os.environ["STRONG_MODEL"]
ROUTER       = "bert"
THRESHOLD    = 0.11593

client = Controller(
    routers=[ROUTER],
    strong_model=STRONG_MODEL,
    weak_model=WEAK_MODEL,
)
