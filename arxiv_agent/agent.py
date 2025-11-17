"""LangGraph A2A conversational agent.

Supports the A2A protocol with messages input for conversational interactions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from openai import AsyncOpenAI

from arxiv_agent.utils.state import State
from langchain.agents import create_agent
from arxiv_agent.utils.tools import search_for_publications_in_arxiv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal


class Context(TypedDict):
    """Context parameters for the agent."""
    my_configurable_param: str


SYSTEM_INSTRUCTION = (
    'You are a specialized assistant for scientific papers search published in Arxiv. '
    "Your sole purpose is to use the 'search_for_publications_in_arxiv' tool to answer questions about scientific papers. "
    "Always include references as URLs from search results to support statements. "
    'If the user asks about anything other than science of engineering, '
    'politely state that you cannot help with that topic and can only assist with science and technology queries. '
    'Do not attempt to answer unrelated questions or use tools for other purposes.'
)


class ResponseFormat(BaseModel):
    """Respond to the user in this format.
    Set response status to input_required if the user needs to provide more information to complete the request.
    Set response status to error if there is an error while processing the request.
    Set response status to completed if the request is complete.
    """

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str

graph  = create_agent(
            model=ChatOpenAI(client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                             model="gpt-4o-2024-11-20"),
            tools=[search_for_publications_in_arxiv],
            system_prompt=SYSTEM_INSTRUCTION,
    response_format=ResponseFormat,
    context_schema=Context,
    # checkpointer=MemorySaver()
    )
