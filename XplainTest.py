import requests
import time

# example model result json

results = """
{
  "status": "finished",
  "whisper": {
    "unseparated_results": {
      "prediction": 0.4408789873123169,
      "label": "Fake"
    },
    "separated_results": {
      "prediction": 0.7842191457748413,
      "label": "Real"
    }
  },
  "rawgat": {
    "unseparated_results": {
      "prediction": 0.45,
      "label": "Fake"
    },
    "separated_results": {
      "prediction": 0.45,
      "label": "Fake"
    }
  },
  "xlsr": {
    "unseparated_results": {
      "prediction": 0.15367034912109376,
      "label": "Fake"
    },
    "separated_results": {
      "prediction": 0.1514015579223633,
      "label": "Fake"
    }
  },
  "vocoder": {
    "unseparated_results": {
      "prediction": 0.6536636352539062,
      "label": "Real"
    },
    "separated_results": {
      "prediction": 0.8053923845291138,
      "label": "Real"
    }
  }
}
"""

AIXPLAIN_API_KEY = "d97f1786ec0701d8809259408d07e4f739f0794bd80a7c30c024004597aff085"
AGENT_ID = "67acb51f56173fdefab4fc62"
POST_URL = f"https://platform-api.aixplain.com/sdk/agents/{AGENT_ID}/run"

headers = {
	"x-api-key": AIXPLAIN_API_KEY,
	"Content-Type": 'application/json'
}

data = {
	"query": "hello",
	# "sessionId": "<SESSIONID_TEXT_DATA>",  # Optional: Specify sessionId from the previous message
}

# POST request to execute the agent
response = requests.post(POST_URL, headers=headers, json=data)
response_data = response.json()
request_id = response_data.get("requestId")

get_url = f"https://platform-api.aixplain.com/sdk/agents/{request_id}/result"

# Polling loop: GET request until the result is completed
while True:
	get_response = requests.get(get_url, headers=headers)
	result = get_response.json()
	
	if result.get("completed"):
		print(result)
		break
	else:
		print("not fininished, sleeping")
		time.sleep(2) # Wait for 5 seconds before checking the result again

# to get results
print(results['data']['output'])
# print("test")
