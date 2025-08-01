from agents import (
    Agent, 
    SQLiteSession, 
    Runner, 
    OpenAIChatCompletionsModel, 
    AgentOutputSchema,
)
from agents.memory.session import SessionABC
from openai.types.responses.response_input_item_param import FunctionCallOutput
from agorm.clients.openai_agents.tools.tool_router import ToolRouter
from agorm.resources.prompt import PromptFactory
from agorm.io import SQLAgentResponse
from openai import AsyncOpenAI
from openai.types.chat_model import ChatModel
from typing import cast
import timeit
import uuid
import os

class Agorm:
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

        self.tool_router = ToolRouter()

    async def search(self, query: str):
        start_time = timeit.default_timer()
        print("Running on query:", query)

        relevant_tools = await self.tool_router.get_tools(query)

        executor_agent = Agent(
            name="ArielSQLExecutor",
            instructions=PromptFactory.get_sql_agent_prompt(),
            model=OpenAIChatCompletionsModel(
                model=self.model,
                openai_client=self.client,
            ),
            tools=relevant_tools,
            output_type=AgentOutputSchema(SQLAgentResponse, strict_json_schema=True),
        )

        response = await Runner.run(
            executor_agent,
            input=query,
            session=self._session,
        )

        print(f"Response: {response.final_output_as(SQLAgentResponse)}")
        elapsed_time = timeit.default_timer() - start_time
        print(f"Elapsed time for SQL Agent: {elapsed_time:.2f} seconds")
        context_history = await self._session.get_items()

        function_call_messages = [function_call for function_call in context_history if function_call.get("type") == "function_call_output"]
        function_call_messages = cast(list[FunctionCallOutput], function_call_messages)
        return [message["output"] for message in function_call_messages]