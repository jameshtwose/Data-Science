# %%
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from glob import glob
# %%
# read in the train, test, and cv data
train_file_list = sorted(glob("data/train*"))
train_data = tf.data.TFRecordDataset(train_file_list)
test_file_list = sorted(glob("data/test*"))
test_data = tf.data.TFRecordDataset(test_file_list)
cv_file_list = sorted(glob("data/val*"))
cv_data = tf.data.TFRecordDataset(cv_file_list)
# %%
# show and example of the (train) data
for record in train_data.take(1):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    print(example)

# %%
cv_df = tfds.as_dataframe(cv_data)
# %%
cv_df.head()

# %%
example

# %%[markdown]
# Define the model architecture
# - Model taken from:
#   - Visser, K., Bosma, B., & Postma, E. (2021). A one-armed CNN for exoplanet detection from light curves. arXiv preprint arXiv:2105.06292.
#   - https://arxiv.org/pdf/2105.06292.pdf
# %%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1, 100, 1)))