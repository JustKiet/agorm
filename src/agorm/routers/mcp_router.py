from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
import faiss  # type: ignore

from agorm.core.interfaces import IDissector
from agorm.core.io import MCPRouterBaseResponse, MCPRouterSSEResponse, MCPRouterStdioResponse, FunctionToolDescription
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from typing import Any, Optional, Literal
from pydantic import SecretStr
from contextlib import AsyncExitStack
import asyncio
import os
import json
import numpy as np
import time

class NormalizedEmbeddings(Embeddings):
    """Wrapper to normalize embeddings for proper cosine similarity."""
    
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs and normalize for cosine similarity."""
        embeddings = self.embeddings.embed_documents(texts)
        return [self._normalize_vector(emb) for emb in embeddings]
    
    def embed_query(self, text: str) -> list[float]:
        """Embed query text and normalize for cosine similarity."""
        embedding = self.embeddings.embed_query(text)
        return self._normalize_vector(embedding)
    
    @staticmethod
    def _normalize_vector(vector: list[float]) -> list[float]:
        """Normalize vector to unit length for cosine similarity."""
        np_vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(np_vector)
        if norm == 0:
            return vector  # Handle zero vector case
        return (np_vector / norm).tolist()
    
class MCPRouter:
    def __init__(
        self,
        *,
        server_url: str | None = None,
        voyageai_api_key: str | None = None,
        voyage_code_embedding_model: str = "voyage-code-3",
        transport: Literal["stdio", "sse"] = "sse",
        dissector: IDissector | None = None,
    ):
        self._server_url = server_url
        if not voyageai_api_key:
            voyageai_api_key = os.getenv("VOYAGEAI_API_KEY")
            if not voyageai_api_key:
                raise ValueError("VOYAGEAI_API_KEY environment variable is not set.")
            
        self.exit_stack = AsyncExitStack()
        
        # Create base embeddings and wrap with normalization for cosine similarity
        base_embeddings = VoyageAIEmbeddings(
            api_key=SecretStr(voyageai_api_key),
            model=voyage_code_embedding_model,
            batch_size=32,
        )
        self.embeddings = NormalizedEmbeddings(base_embeddings)
        self._transport = transport

        # Use IndexFlatIP with normalized vectors for cosine similarity
        # Cosine similarity = dot product of normalized vectors
        embedding_dim = len(self.embeddings.embed_query("test"))
        index = faiss.IndexFlatIP(embedding_dim)
        
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        if dissector is None:
            from agorm.clients.openai.dissector import OpenAIDissector
            dissector = OpenAIDissector()
        
        self.dissector = dissector
        self._total_schemas = 0
        self.schemas: list[dict[str, Any]] = []
        self.stdio_server_params: Optional[StdioServerParameters] = None

    @classmethod
    async def from_sse(
        cls, 
        server_url: str,
        dissector: IDissector | None = None,
        voyageai_api_key: str | None = None,
        voyage_code_embedding_model: str = "voyage-3-large",
    ) -> "MCPRouter":
        # Use proper async context management
        async with sse_client(server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                res = await session.list_tools()
                tools = res.tools

                cls.schemas: list[dict[str, Any]] = [tool.model_dump() for tool in tools]

                instance = cls(
                    server_url=server_url,
                    voyageai_api_key=voyageai_api_key,
                    voyage_code_embedding_model=voyage_code_embedding_model,
                    transport="sse",
                    dissector=dissector,
                )
                cls._total_schemas = len(cls.schemas)
                await instance.store_schemas(cls.schemas)

                return instance
            
    @classmethod
    async def from_stdio(
        cls,
        server_script_path: str,
        dissector: IDissector | None = None,
        env: dict[str, str] | None = None,
        voyageai_api_key: str | None = None,
        voyage_code_embedding_model: str = "voyage-3-large",
    ) -> "MCPRouter":
        is_python = False
        is_js = False
        command = None
        args = [server_script_path]

        # Check if the server script is a path or a package
        if server_script_path.startswith("@") or "/" not in server_script_path:
            is_js = True
            command = "npx"
        else:
            # Is a file path
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a Python or JavaScript file.")
            command = "python" if is_python else "node"

        cls.stdio_server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        # Start the server
        async with stdio_client(cls.stdio_server_params) as (stdio, writer):
            async with ClientSession(stdio, writer) as session:
                await session.initialize()

                res = await session.list_tools()
                tools = res.tools
                cls.schemas: list[dict[str, Any]] = []

                for tool in tools:
                    cls.schemas.append(tool.model_dump())

                instance = cls(
                    server_url=None,
                    voyageai_api_key=voyageai_api_key,
                    voyage_code_embedding_model=voyage_code_embedding_model,
                    transport="stdio",
                    dissector=dissector,
                )
                cls._total_schemas = len(cls.schemas)
                await instance.store_schemas(cls.schemas)

                return instance
            
    async def store_schemas(self, schemas: list[dict[str, Any]]) -> None:
        """
        Store the schemas in a retriever for later use.
        """
        print(f"Storing {len(schemas)} schemas in the vector store.")
        
        self.vector_store.add_texts( # type: ignore[call-arg]
            texts=[
                json.dumps({
                    "function_name": schema["name"],
                    "description": schema["description"],
                })
                for schema in schemas
            ],
        )

    async def get_relevant_tools(self, query: str) -> list[str]:
        """
        Retrieve the most relevant tools based on the query.

        :param str query: The query to search for relevant tools.
        :return: A list of relevant tool names or an empty list if no tools are found.
        :rtype: list[str]
        """
        start_time = time.time()
        results = await self.vector_store.asimilarity_search_with_score( # type: ignore[call-arg]
            query=f"I need to find a function that can answer this query: {query}",
            k=1,
        )

        if not results:
            return []

        print(f"Found {len(results)} results for query: {query}")
        
        # Create a set of valid tool names for O(1) lookup
        valid_tool_names = {tool["name"] for tool in self.schemas}
        
        tool_info: list[str] = []
        for result, _ in results:
            tool_schema = json.loads(result.page_content)
            function_name = tool_schema["function_name"]
            if function_name in valid_tool_names:
                tool_info.append(function_name)
            else:
                print(f"Tool {function_name} not found in schemas.")
        
        print(f"Total time taken for tool retrieval: {time.time() - start_time:.2f} seconds")

        return tool_info
    
    async def route_tools(
        self,
        query: str,
    ) -> Optional[MCPRouterBaseResponse]:
        """
        Route the tools based on the actionable steps.

        :param str query: The query to route tools for.
        :return: An MCPServer instance with the routed tools.
        :rtype: Optional[MCPServer]
        """
        start_time = time.time()
        actionable_steps = await self.dissector.dissect_query(
            query=query,
            tool_descriptions=[
                FunctionToolDescription(function_name=tool["name"], description=tool["description"])
                for tool in self.schemas
            ]
        )
        
        if not actionable_steps:
            print("No actionable steps to route.")
            return None

        print(f"Routing tools for actionable steps: {actionable_steps}")
        tasks = [self.get_relevant_tools(subquery) for subquery in actionable_steps]
        tools_results = await asyncio.gather(*tasks)
        required_tools = list(set(tool for tools in tools_results for tool in tools))

        if not required_tools:
            print("No relevant tools found.")
            return None
        
        print(f"Required tools after routing: {required_tools}")
        print(f"Total time taken for routing: {time.time() - start_time:.2f} seconds")

        if self._server_url and self._transport == "sse":
            return MCPRouterSSEResponse(
                transport_type="sse",
                tool_names=required_tools,
                actionable_steps=actionable_steps,
                server_url=self._server_url,
            )
        elif self.stdio_server_params and self._transport == "stdio":
            return MCPRouterStdioResponse(
                transport_type="stdio",
                tool_names=required_tools,
                actionable_steps=actionable_steps,
                stdio_params=self.stdio_server_params,
            )
        else:
            return None
        