from agorm.clients.openai_agents.agent import Agorm
from dotenv import load_dotenv

load_dotenv()

agorm = Agorm()

async def main():
    query = "Get me everything about David"
    response = await agorm.search(query)
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())