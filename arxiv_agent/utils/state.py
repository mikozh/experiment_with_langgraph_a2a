from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure for A2A conversational messages.
    """
    messages: List[Dict[str, Any]]
