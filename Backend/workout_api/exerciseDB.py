#%%
import requests
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from ast import literal_eval

#%%
load_dotenv(find_dotenv())

#%%
url = "https://exercisedb.p.rapidapi.com/exercises/equipment/body%20weight"

headers = {
    'x-rapidapi-host': "exercisedb.p.rapidapi.com",
    'x-rapidapi-key': os.getenv("RAPID_API_KEY")
    }

#%%
response = requests.request("GET", url, headers=headers)

# %%
df = pd.DataFrame(literal_eval(response.text))
# %%
df.head()
# %%
