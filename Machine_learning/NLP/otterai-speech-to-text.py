# %%[markdown]
# The following uses otterai from here: 
# `pip install git+https://github.com/chadlohrli/otterai-api.git`

#%%
from dotenv import load_dotenv, find_dotenv
import os
from otterai import OtterAI
#%
otter = OtterAI()
otter.login(os.environ["otterai_username"], 
            os.environ["otterai_password"])
# %%
otter.get_user()
# %%
[x["summary"] for x in otter.get_speeches()["data"]["speeches"]]
[x["speech_id"] for x in otter.get_speeches()["data"]["speeches"]]
# %%
otter.get_speech(speech_id="26QJCCGHA6U36L5V")
# %%
