from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.retrievers import KNNRetriever

from agents import Tool

from agorm.clients.openai_agents.tools.tools import get_user_info, get_job_info, get_company_info

from pydantic import SecretStr
from enum import Enum
import json
import os

class ToolRegistry(str, Enum):
    GET_USER_INFO = "get_user_info"
    GET_JOB_INFO = "get_job_info"
    GET_COMPANY_INFO = "get_company_info"
    
class ToolRouter:
    def __init__(
        self,
        voyageai_api_key: str | None = None,
        voyage_code_embedding_model: str = "voyage-code-3",
    ):
        if not voyageai_api_key:
            voyageai_api_key = os.getenv("VOYAGEAI_API_KEY")
            if not voyageai_api_key:
                raise ValueError("VOYAGEAI_API_KEY environment variable is not set.")
            
        self.embeddings = VoyageAIEmbeddings(
            api_key=SecretStr(voyageai_api_key),
            model=voyage_code_embedding_model,
            batch_size=32,
        )
        self.tools = {
            ToolRegistry.GET_USER_INFO: get_user_info,
            ToolRegistry.GET_JOB_INFO: get_job_info,
            ToolRegistry.GET_COMPANY_INFO: get_company_info,
        }
        self.schemas = [
            json.dumps(
                {
                    "tool_name": tool.name,
                    "tool_description": tool.description,
                    "args_schema": tool.params_json_schema
                }
            )
            for tool in self.tools.values()
        ]
        self.retriever = KNNRetriever.from_texts( # type: ignore[call-arg]
            texts=self.schemas,
            embeddings=self.embeddings,
            k=3,
        )

    async def get_tools(self, query: str) -> list[Tool]:
        """
        Retrieve the most relevant tool based on the query.
        """
        results = await self.retriever.ainvoke(query)
        if not results:
            return []

        tools: list[Tool] = []
        for result in results:
            tool_schema = json.loads(result.page_content)
            tool = self.tools.get(ToolRegistry(tool_schema["tool_name"]))
            if tool is None:
                return []
            tools.append(tool)

        print(f"Retrieved tools: {[tool.name for tool in tools]}")

        return tools

