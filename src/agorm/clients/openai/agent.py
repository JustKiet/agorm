from agents import (
    Agent, 
    SQLiteSession, 
    Runner, 
    OpenAIChatCompletionsModel, 
    AgentOutputSchema,

)
from agents.mcp import (
    create_static_tool_filter,
    MCPServerSse, 
    MCPServerStdio, 
    MCPServerSseParams, 
    MCPServerStdioParams
)
from agents.memory.session import SessionABC
from openai.types.responses.response_input_item_param import FunctionCallOutput
from agorm.routers.mcp_router import MCPRouter
from agorm.resources.prompt import PromptFactory
from agorm.core.io import SQLAgentResponse, MCPRouterSSEResponse, MCPRouterStdioResponse
from agorm.core.interfaces import IAgorm
from openai import AsyncOpenAI
from openai.types.chat_model import ChatModel
from typing import cast
import timeit
import uuid
import os

class OpenAIAgorm(IAgorm):
    def __init__(
        self,
        session: SessionABC | None = None,
        client: AsyncOpenAI | None = None,
        model: ChatModel = "gpt-4o-mini",
    ) -> None:
        if not session:
            session_id = str(uuid.uuid4())
            session = SQLiteSession(session_id=session_id)

        self._session_id = session.session_id
        self._session = session
        self.model = model

        if not client:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            
            client = AsyncOpenAI(
                api_key=api_key,
            )

        self.client = client

    async def initialize_sse(self, server_url: str) -> None:
        self.mcp_router = await MCPRouter.from_sse(
            server_url=server_url,
        )

    async def search(self, query: str) -> list[str]:
        start_time = timeit.default_timer()
        print("Running on query:", query)
        
        route_res = await self.mcp_router.route_tools(query)

        if not route_res:
            print("No tools routed for the query.")
            return []

        filtered_mcp_server = None
        if isinstance(route_res, MCPRouterSSEResponse):
            filtered_mcp_server = MCPServerSse(
                params=MCPServerSseParams(
                    url=route_res.server_url,
                ),
                tool_filter=create_static_tool_filter(
                    allowed_tool_names=route_res.tool_names if route_res.tool_names else None,
                )
            )
        elif isinstance(route_res, MCPRouterStdioResponse):
            filtered_mcp_server = MCPServerStdio(
                params=MCPServerStdioParams(
                    command=route_res.stdio_params.command,
                    args=route_res.stdio_params.args,
                    env=route_res.stdio_params.env if route_res.stdio_params.env else {},
                ),
                tool_filter=create_static_tool_filter(
                    allowed_tool_names=route_res.tool_names if route_res.tool_names else None,
                )
            )

        if not filtered_mcp_server:
            print("No relevant tools found for the query.")
            return []

        async with filtered_mcp_server as mcp_server:
            executor_agent = Agent(
                name="AgormSQLExecutor",
                instructions=PromptFactory.get_sql_agent_prompt(),
                model=OpenAIChatCompletionsModel(
                    model=self.model,
                    openai_client=self.client,
                ),
                mcp_servers=[mcp_server],
                output_type=AgentOutputSchema(SQLAgentResponse, strict_json_schema=True),
            )

            response = await Runner.run(
                executor_agent,
                input=query + f"Here are the actions you should take based on the tools provided: {route_res.actionable_steps}",
                session=self._session,
            )

            print(f"Response: {response.final_output_as(SQLAgentResponse)}")
            elapsed_time = timeit.default_timer() - start_time
            print(f"Elapsed time for SQL Agent: {elapsed_time:.2f} seconds")
            context_history = await self._session.get_items()

            function_call_messages = [function_call for function_call in context_history if function_call.get("type") == "function_call_output"]
            function_call_messages = cast(list[FunctionCallOutput], function_call_messages)
            return [message["output"] for message in function_call_messages]