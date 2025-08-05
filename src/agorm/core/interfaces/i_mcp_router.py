from typing import Protocol

class IMCPRouter(Protocol):
    async def route_tools(self, query: str) -> list[str]:
        ...