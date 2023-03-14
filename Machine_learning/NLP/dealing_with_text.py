# %%[markdown]
# ### Manipulating text

# Text collected from here:
# - https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
# - https://www.aaai.org/Papers/Symposia/Spring/2006/SS-06-03/SS06-03-039.pdf
# - Schler, J., Koppel, M., Argamon, S., & Pennebaker, J. W. (2006, March). Effects of age and gender on blogging. In AAAI spring symposium: Computational approaches to analyzing weblogs (Vol. 6, pp. 199-205).

# %%
from tqdm import tqdm
import pandas as pd
from glob import glob
from bs4 import BeautifulSoup as bs
from jmspack.utils import flatten
# %%
files_list = glob("blogs/*")#[0:500]
files_list[2]

# %%
all_ngrams_df = pd.DataFrame()
for file in tqdm(files_list):
    with open(file, "r", encoding = "ISO-8859-1") as file:
        # Read each line in the file, readlines() returns a list of lines
        content = file.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        bs_content = bs(content, "lxml")
        text_list = bs_content.find_all("post")
        prepped_list = [text.get_text().replace("\n", "").replace("\t", "") for text in text_list]

        all_ngrams_df = pd.concat([all_ngrams_df, 
                                   get_ngrams(text=prepped_list, 
                                              ngram_length_tuple=(5, 5)).iloc[50:99, :]], 
                                  axis=0)

# %%
five_grams_df = (all_ngrams_df
                 .groupby("trigram")
                 .sum()
                 .reset_index()
                 .sort_values(by="frequency", ascending=False)
                 .head(1000)
                 .sample(n=60, random_state=42)
                 .sort_values(by="frequency", ascending=False)
                 )
five_grams_df
# %%
trigrams_df.to_csv("data/blogs_50_to_99_head_1000_sample_50_random_state_42_trigrams.csv")
four_grams_df.to_csv("data/blogs_50_to_99_head_1000_sample_50_random_state_42_four_grams.csv")
five_grams_df.to_csv("data/blogs_50_to_99_head_1000_sample_50_random_state_42_five_grams.csv")

# %%
trigrams_df = pd.read_csv("data/blogs_50_to_99_head_1000_sample_50_random_state_42_trigrams.csv", index_col=0)
four_grams_df = pd.read_csv("data/blogs_50_to_99_head_1000_sample_50_random_state_42_four_grams.csv", index_col=0)
five_grams_df= pd.read_csv("data/blogs_50_to_99_head_1000_sample_50_random_state_42_five_grams.csv", index_col=0)
# %%
five_grams_df.reset_index(drop=True).to_dict()
# %%
