from typing import Protocol
from agorm.core.io import FunctionToolDescription

class IDissector(Protocol):
    async def dissect_query(
        self, 
        query: str,
        tool_descriptions: list[FunctionToolDescription]
    ) -> list[str]:
        """
        Dissect the query into actionable steps to identify relevant tools.

        :param str query: The query to dissect.
        :param list[FunctionToolDescription] tool_descriptions: List of available tools and their descriptions for context.
        :return: list of tool names that are relevant to the query.
        :rtype: list[str]
        """
        ...