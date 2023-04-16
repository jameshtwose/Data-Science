# %%
from google_alerts import GoogleAlerts
# %%
# Create an instance
ga = GoogleAlerts(email="james.twose2711@gmail.com", 
                  password="B0ringBanal1ty")

# Authenticate your user
ga.authenticate()

# %%
# List configured monitors
ga.list()

# %%
# Add a new monitor
ga.create("Hello World", {"delivery": "RSS"})

# %%
# Modify an existing monitor
ga.modify(
    "89e517961a3148c7:c395b7d271b4eccc:com:en:US", {"delivery": "RSS", "monitor_match": "ALL"}
)

# %%
# Delete a monitor
ga.delete("89e517961a3148c7:c395b7d271b4eccc:com:en:US")
