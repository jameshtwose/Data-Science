# %%
from oauthlib.oauth2 import WebApplicationClient
import requests
# %%
client_id = "23BDP7"
client_secret = "f1acd4b5730f4ccc51f5e9d5bc6acf56"
client = WebApplicationClient(client_id)

# %%
authorization_url = "https://www.fitbit.com/oauth2/authorize"

url = client.prepare_request_uri(
    authorization_url,
    redirect_uri="http://127.0.0.1:8000/login/fitbit/authorized/",
    scope=["read:user"],
    #   state = 'D8VAo311AAl_49LAtM51HA'
)

# %%
data = client.prepare_request_body(
    #   code = 'yyyyyyy',
    redirect_uri="http://127.0.0.1:8000/login/fitbit/authorized/",
    client_id=client_id,
    client_secret=client_secret,
)

# %%
data
# %%
