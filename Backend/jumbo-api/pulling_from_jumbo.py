#%%
import requests
import json
import pandas as pd
import numpy as np

# %%
BASE_URL = "https://mobileapi.jumbo.com/v14/products/"
PRODUCT_ID="478570DS"
# PRODUCT_ID="144567PAK"
url = f"{BASE_URL}{PRODUCT_ID}"

#%%
response = requests.request("GET", url, 
                            # headers=headers
                            )
# %%
output_dict=json.loads(response.text)["product"]["data"]
nutrients_dict=output_dict["nutritionalInformation"][0]["nutritionalData"]["entries"]
food_df = (pd.DataFrame(nutrients_dict, 
             index=np.repeat(output_dict["title"], 
                             repeats=len(nutrients_dict)))
 .reset_index()
 .rename(columns={"index": "food"})
 )

#%%
food_df
# %%
