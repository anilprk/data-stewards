# Imports
import requests
from typing import List, Optional
from pydantic import BaseModel
from perplexity import Perplexity

# API Client
client = Perplexity()

# Pydantic Data Model
class HCPData(BaseModel):
    NPI: int
    street: str
    city: str
    state: str
    country: str
    degrees: list[str]
    contact_details: list[str]

# Function to get hcp data in structured format
def get_details_for_hcp(hcp_name, model_name='sonar', should_use_pro_search=False):
    user_query = f"Give me the NPI, Street, City, State, Province, Zipcode, Degree(s) of the health care provider in us named {hcp_name}"
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": user_query
            }
        ],
        response_format= {
            "type": "json_schema",
            "json_schema": {
                "schema": HCPData.model_json_schema()
            }
        }
    )
    print(f"Total Cost for the request is: {completion.usage.cost.total_cost}")
    # data = HCPData.model_validate_json(completion.choices[0].message.content)
    data = completion.choices[0].message.content
    return data
