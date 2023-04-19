# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from bertopic import BERTopic
from transformers import AutoTokenizer, BartForConditionalGeneration, pipeline
from textblob import TextBlob
from jmspack.utils import flatten

# from nltk.corpus import stopwords

plt.style.use("jms_style_sheet")
# %%
# read in stop words
with open("data/stop_words.txt", "r") as f:
    stopwords = [x.replace("\n", "") for x in f.readlines()[1:]]

# %%
# "summarized_text" column
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    framework="pt",
)


# %%
# "adjectives_text" column
def get_adjectives(text):
    blob = TextBlob(text)
    return [word for (word, tag) in blob.tags if tag == "JJ"]


# %%
df = (
    pd.read_json(path_or_buf="data/train_reviews.json").assign(
        **{
            # reassigning the stars column to be an integer
            "stars": lambda x: x["stars"].str.extract("(\d+)").astype(int),
            # creating a "prepped" version of the text column by removing stopwords
            # and converting to lowercase
            "stopwords_removed_text": lambda x: x["text"]
            .str.lower()
            .str.replace(
                # pat="$|".join([x for x in stopwords if len(x) > 2]),
                pat="$|".join(stopwords),
                repl="",
                regex=True,
            ),
            # "summarized_text": lambda d: d.apply(
            #     lambda x: summarizer(x["text"], min_length=5, max_length=20)[0][
            #         "summary_text"
            #     ],
            #     axis=1,
            # ),
            "adjectives_text": lambda d: d.apply(
                lambda x: ", ".join(get_adjectives(x["text"])), axis=1
            ),
        }
    )
    # when looking at the data initially, I noticed that there are some reviews
    # that are complete duplicates across the title and text columns. I decided to drop these
    # to avoid any over representation of these reviews
    .drop_duplicates(subset=["title", "text"])
    # following the row dropping reset the index to be sequential
    .reset_index(drop=True)
)
df.head()

# %%
df.info()
# %%
_ = sns.countplot(x="stars", data=df)
# %%
_ = sns.lineplot(data=df, x="date", y="stars")
_ = sns.scatterplot(
    data=df, x="date", y="stars", edgecolor="grey", linewidth=2, zorder=1
)

# %%
count_cutoff = 20
adjectives_df = (
    df["adjectives_text"]
    .str.split(",", expand=True)
    .stack()
    .reset_index()
    .drop(["level_0", "level_1"], axis=1)
    .rename(columns={0: "adjective"})
    .value_counts()
    .reset_index()
    .rename(columns={"index": "adjective", 0: "count"})
    .loc[lambda d: d["count"] > count_cutoff]
    .loc[lambda d: ~d["adjective"].isin(["", " ’"])]
)
_ = plt.figure(figsize=(5, 8))
_ = sns.barplot(data=adjectives_df, y="adjective", x="count")
_ = plt.title(f"Adjectives with a count > {count_cutoff}")
# %%
df["date"].min(), df["date"].max()
# %%
df["title"].value_counts()

# %%
title_counts = df["title"].value_counts()
top_titles = title_counts[title_counts > 2].index.tolist()
len(top_titles)
# %%
# select and visualize the reviews that have a title that is duplicated
df[df["title"].isin(top_titles)].sort_values("title")
# %%
# it would appear that a lot of the reviews are directed at
# explaining how awful/ horrible/ terrible the experience was
# and specifically the service. It is likely that these will be
# the main topics that come out of the topic extraction (especially
# in the 1 star reviews)
# %% [markdown]
# ### Topic Extraction
# The following methods will be auditioned for topic extraction:
# - spacy
# - spacy with bertopic
# - gensim LDAModel

# %%[markdown]
#### spacy with bertopic
# %%
column_string = "text"
# column_string = "title"
# column_string = "adjectives_text"
low_docs = df.loc[lambda d: d["stars"] == 1, column_string].tolist()
hi_docs = df.loc[lambda d: d["stars"] >= 3, column_string].tolist()
docs = df[column_string].tolist()
len(low_docs), len(hi_docs), len(docs)
# %%
# load the spacy model
# python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_md")
topic_model = BERTopic(
    embedding_model=nlp, language="english", calculate_probabilities=True
)
# %%
for doc_version in [
    docs,
    # low_docs,
    # hi_docs
]:
    topics, probs = topic_model.fit_transform(doc_version)
    topics_list = [
        x[0]
        for x in flatten(
            [topic_model.get_topic(topic) for topic in topic_model.get_topics().keys()]
        )
        if len(x[0]) > 1
    ]
    print(topics_list)
    fig = topic_model.visualize_topics()
    fig.show()
# %%
# seems like the topics that come up are for the most part negative or
# generic stopwords. Thus it seems like we should probably remove them.
# This is done in the "prepped_text" column

# %%
pd.DataFrame(
    [topic_model.get_topic(topic) for topic in topic_model.get_topics().keys()]
)
# %%
topics_list
# %%
len(doc_version)
# %%
