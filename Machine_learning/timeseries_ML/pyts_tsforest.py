# %%
from pyts.datasets import load_gunpoint
from pyts.classification import TimeSeriesForest
import pandas as pd
# %%
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
clf = TimeSeriesForest(random_state=43)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
# %%
# X_train
pd.DataFrame(X_train)
# %%
pd.DataFrame(y_train)
# %%
