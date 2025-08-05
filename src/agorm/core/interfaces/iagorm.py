from typing import Protocol

class IAgorm(Protocol):
    async def search(self, query: str) -> list[str]:
        ...