from agorm.core.interfaces import IDissector
from agorm.core.io import ActionableSteps, FunctionToolDescription
from openai import AsyncOpenAI
from openai.types.chat_model import ChatModel
import json
import os

class OpenAIDissector(IDissector):
    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        instructions: str = """You are an expert strategic planner. 
        Given a query, dissect it into subactions (actionable steps) that can be performed to complete the query.
        Each subaction should be a simple, single, actionable task that can be performed by a tool.
        For example, if the query is "I need to find a candidate in IT field and their expertise",
        the subactions could be:
        - "Get all available expertises (to get the IT expertise ID)"
        - "Search for candidates in IT field"
        - "Get expertise of candidates"
        """,
        model: ChatModel = "gpt-4o-mini",
        temperature: float = 0.5,
    ) -> None:
        if not client:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            
            client = AsyncOpenAI(
                api_key=api_key,
            )
        self.client = client
        self.model = model
        self.instructions = instructions
        self.temperature = temperature
        
    async def dissect_query(
        self,
        query: str,
        tool_descriptions: list[FunctionToolDescription]
    ) -> list[str]:
        completion = await self.client.chat.completions.parse(
            messages=[
                {
                    "role": "system",
                    "content": self.instructions
                },
                {
                    "role": "user",
                    "content": f"""
                    Dissect the following query into subqueries (actions) that can be performed by tools: {query}
                    Here are all available tools and their descriptions for context:
                    {json.dumps([{"function_name": tool.function_name, "description": tool.description} for tool in tool_descriptions], indent=2)}
                    """
                }
            ],
            model=self.model,
            temperature=self.temperature,
            response_format=ActionableSteps,
        )

        actionable_steps = completion.choices[0].message.parsed

        if actionable_steps is None or not actionable_steps.steps:
            print("No actionable steps found.")
            return list(query)

        print(f"Actionable steps: {actionable_steps.steps}")
        return actionable_steps.steps
