from pydantic import BaseModel
from enum import Enum

class Action(Enum):
    FINISHED = "finished"
    CALL_MORE_TOOLS = "call_more_tools"
    NOT_FOUND = "not_found"

class SQLAgentResponse(BaseModel):
    next_action: Action