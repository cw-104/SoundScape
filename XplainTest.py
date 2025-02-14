## Create an Agent
import os
os.environ["AIXPLAIN_API_KEY"] = "9ee1be3db852983c78004eebf5172e6688918b10366edd67e33d7095e1ed18da"

from aixplain.factories import AgentFactory
from aixplain.modules.agent import ModelTool

agent = AgentFactory.create(
	name="Google Search agent",
	description="You are an agent that uses Google Search to answer queries.",
	tools=[
		# Google Search
		ModelTool(model="65c51c556eb563350f6e1bb1"),
	],
)
agent_response = agent.run("What's an AI agent?")