class PromptFactory:
    @staticmethod
    def get_sql_agent_prompt() -> str:
        with open("/Users/justkiet/Coding/sql-agent/src/agorm/resources/system_message_v2.xml", "r") as file:
            return file.read()

