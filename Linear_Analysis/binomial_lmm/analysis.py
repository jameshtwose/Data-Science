# %%
import os

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
# %%
import pandas as pd
import numpy as np
from pymer4.models import Lmer
from pymer4.stats import discrete_inverse_logit
import seaborn as sns
from sklearn.metrics import confusion_matrix

# %%
# good reference: https://library.virginia.edu/data/articles/getting-started-with-binomial-generalized-linear-mixed-models

# %%
# create nested data with 7 rows per subject and 3 columns of predictors and with 1 column of boolean outcome
np.random.seed(123)
n = 100
subject_list = list(np.repeat(range(n), 7))
subject_to_outcome_dict = dict(zip(subject_list, np.random.randint(0, 2, size=n * 7)))
# %%
pre_df = pd.DataFrame(
    {
        "days": np.tile(range(7), n),  # "days" is the "time" variable
        "subject": subject_list,
        "x1": np.random.normal(size=n * 7),
        "x2": np.random.normal(size=n * 7),
        "x3": np.random.normal(size=n * 7),
        # outcome is either 0 or 1 and does not change within subject
        "outcome": np.repeat([subject_to_outcome_dict[i] for i in subject_list], 1),
    }
)
pre_df.shape

# %%
# add a slope for each subject to the x1 predictor
# df["x1"] = df["x1"] + df["subject"] * 0.5
# add a slope for each subject to the x2 predictor that is 0 for outcome==0 and 1 for outcome==1
# df["x2"] = df["x2"] * df["outcome"] * 1.5

df = pre_df.assign(
    **{
        # "x1": lambda x: x["x1"] + x["subject"] * 0.5,
        "x2": lambda x: 0.2 + x["x2"] * x["outcome"] * 1.5,
    }
)

# %%
df.head()
# %%
_ = sns.scatterplot(
    x="subject",
    y="outcome",
    data=df,
)
# %%
plot_df = df.melt(id_vars=["subject", "outcome", "days"], value_vars=["x1", "x2", "x3"])

# %%
_ = sns.lmplot(
    x="days",
    y="value",
    hue="subject",
    col="outcome",
    row="variable",
    data=plot_df,
    ci=None,
    legend=False,
    sharey=False,
)

# %%
model = Lmer(
    "outcome ~ days + x1 + x2 + x3 + (days | subject)", data=df, family="binomial"
)
# %%
results = model.fit()
results
# %%
_ = model.plot_summary()
# %%
# get the inverse logits of the fixed effects
discrete_inverse_logit(results.loc["(Intercept)","Estimate"])
# %%
for predictor in results.index.tolist()[1:]:
    intercept = results.loc[predictor,"Estimate"] + results.loc["(Intercept)","Estimate"]
    inverse_logit = discrete_inverse_logit(intercept)
    print(f"{predictor} inverse logit: {inverse_logit}")
# %%
y_pred = model.predict(data=df, verify_predictions=False)
# %%
true_pred_df = pd.DataFrame(
    {
        "subject": df["subject"],
        "outcome": df["outcome"],
        "y_pred": y_pred,
    }
)
# %%
_ = sns.lmplot(
    x="outcome",
    y="y_pred",
    data=true_pred_df,
)
# %%
conf_mat = confusion_matrix(y_true = true_pred_df["outcome"], 
                 y_pred = true_pred_df["y_pred"] > 0.5)
# %%
_ = sns.heatmap(conf_mat, annot=True, fmt='g')
# %%
