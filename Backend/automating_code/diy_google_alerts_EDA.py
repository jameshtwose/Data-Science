# %%[markdown]
# I set up a google alert for rubber pricing - pushing alerts to an RSS feed.
# The RSS feed is then scraped and the data is stored in a pandas dataframe.
# %%
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# %%
rss_feed = "https://www.google.com/alerts/feeds/17740685286653827239/3974289959828158708"
# %%
data = requests.get(rss_feed).text
# %%
# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")
# %%
# Finding all instances of tag
# `title`
b_title = Bs_data.find_all("title")
title_list = [
    title.text.replace("<b>", "").replace("</b>", "").replace("&amp", "and") for title in b_title
]
title_list
# %%
# Finding all instances of tag
# `link`
b_link = Bs_data.find_all("link")
links_list = [str(link).split("&amp;url=")[-1] for link in b_link]
# %%
# Finding all instances of tag
# `published`
b_published = Bs_data.find_all("published")
published_list = [pub.text for pub in b_published]
# %%
df = pd.DataFrame(
    {
        "title": title_list,
        "link": links_list,
        "published": [datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")] + published_list,
    }
).assign(**{"published": lambda x: pd.to_datetime(x.published)})
# %%
html_page = requests.get(df.loc[1, "link"]).text
# %%
link_text = BeautifulSoup(html_page, 'html.parser').find_all(string=True)
# %%
output = ''
blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head', 
    'input',
    'script',
    # there may be more elements you don't want, such as "style", etc.
]

for t in link_text:
    if t.parent.name not in blacklist:
        output += '{} '.format(t)

print(output)
# %%
