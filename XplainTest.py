## Create an Agent
import os
os.environ["AIXPLAIN_API_KEY"] = "9ee1be3db852983c78004eebf5172e6688918b10366edd67e33d7095e1ed18da"

#@title # **Create and run your first AI Agent**

#@markdown AI agents 🤖 are software that use reasoning models, like **language models** (LLM/SLM), and **tools** like APIs or custom code, along with data sources, to reason, plan, reflect, and execute tasks **autonomously**. With some tweaking, they can also learn and self-improve!

#@markdown **Name your agent and describe its role and task. The more precise your task description, the better your agent will perform.**

Name = "Weather Agent" #@param {type:"string"}
Instructions = "An agent that answers queries about the current weather." #@param {type:"string"}

#@markdown **Add a tool to your agent. This is the tool ID of the [Open Weather API](https://platform.aixplain.com/discover/model/66f83c216eb563266175e201). Find more in aiXplain's [marketplace](https://platform.aixplain.com/discover).**

Tool = "66f83c216eb563266175e201" #@param {type:"string"}
Format = "The input query of this tool must be of the form 'text': 'City'." #@param {type:"string"}

from aixplain.factories import AgentFactory
from aixplain.modules.agent.tool.model_tool import ModelTool

agent = AgentFactory.create(
	name=Name,
	description = "Agent for weather detection",
	instructions=Instructions,
	tools=[
		ModelTool(model=Tool, description=Format),
	],
) # default LLM is GPT4o


#@markdown **Run and test your agent**

Query = "What is the weather in Liverpool, UK?" #@param {type:"string"}

agent_response = agent.run(Query)
agent_response['data']['output']

#agent.deploy() #to deploy your agent on aiXplain's platform.

#@markdown **To learn how to debug your agent, build a team agent, and deploy your agent, follow along.**

#@markdown **Ask a follow up questions**

Query = "Is that too cold for shorts?" #@param {type:"string"}
session = agent_response["data"]["session_id"]
agent_response = agent.run(Query,session_id=session)
agent_response['data']['output']