from aixplain.factories import TeamAgentFactory

team = TeamAgentFactory.create(
    name="Wiki and Web Team Agent",
    description="You take user queries and search them using wiki or by web scraping URLs if appropriate.",
    agents=[scraper_agent, wiki_agent],
    llm_id="6646261c6eb563165658bbb1" # GPT-4o
)

# print(team.name, team.id)

team.__dict__ # Team agent attributes

result = team.run("Tell me about aiXplain. They have a website, aixplain.com.") # run team agent
print(result['data']['output'])