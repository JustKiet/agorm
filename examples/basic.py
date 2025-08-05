from agorm.clients.openai.agent import OpenAIAgorm
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
import os
import time

load_dotenv()

agorm = OpenAIAgorm()

@function_tool
async def search_sql(query: str) -> list[str]:
    """
    Search for SQL queries related to the given query string. Takes in a query string (intend - what you want to query for) in natural language and returns a list of SQL query results.

    :param str query: The natural language query to search for.
    :return: A list of SQL query results.
    :rtype: list[str]
    :example: search_sql("What are the top 10 most expensive products?")
    """
    response = await agorm.search(query)
    print(f"Search results: {response}")
    return response

agent = Agent(
    name="SQLAgent",
    instructions="You are a SQL agent that can execute SQL queries based on natural language input. Use the search_sql tool to find relevant SQL queries. When using the tool, make sure to provide as much details as possible to ensure accurate results.",
    tools=[search_sql],
    model=OpenAIChatCompletionsModel(
        model="gpt-4.1-mini",
        openai_client=AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    ),
)

async def main():
    print("Running test for SQL agent...")
    await agorm.initialize_sse(server_url="http://localhost:8001/sse")
    start = time.time()
    query = "Get all candidates that has name 'John' and works in 'Healthcare' field."
    print(f"Running agent with query: {query}")
    res = Runner.run_streamed(agent, input=query)
    track_first_token = False
    async for event in res.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            if not track_first_token:
                time_to_first_token = time.time() - start
                print(f"Time to first token: {time_to_first_token:.2f} seconds")
                track_first_token = True
            print(event.data.delta, end="", flush=True)

    print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())