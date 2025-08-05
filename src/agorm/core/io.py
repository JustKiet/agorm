from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass
from typing import Literal, TypeVar
from mcp import StdioServerParameters

class Action(Enum):
    FINISHED = "finished"
    CALL_MORE_TOOLS = "call_more_tools"
    NOT_FOUND = "not_found"

class SQLAgentResponse(BaseModel):
    next_action: Action

T = TypeVar("T")

@dataclass
class MCPRouterBaseResponse:
    tool_names: list[str]
    actionable_steps: list[str]

@dataclass
class MCPRouterSSEResponse(MCPRouterBaseResponse):
    transport_type: Literal["sse"]
    server_url: str

@dataclass
class MCPRouterStdioResponse(MCPRouterBaseResponse):
    transport_type: Literal["stdio"]
    stdio_params: StdioServerParameters

