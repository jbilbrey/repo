
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#dataset_path

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# Clean the data - The dataset contains a few unknown values.  Drop these values.
#print("Here is the sum of the na values... " + str(dataset.isna().sum()))
print(dataset.isna().sum())
dataset = dataset.dropna()

# The "Origin" column is really categorical, not numeric. So convert that to a one-hot:
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

# We will use the test set in the final evaluation of our model.

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
# Have a quick look at the joint distribution of a few pairs of columns from the training set.
#sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# Also look at the overall statistics:
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print("---------------------------------------------------------------------------")
print(train_stats)

# Split features from labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalize the data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# check the results of normalization
train_stats2 = normed_train_data.describe()
train_stats2 = train_stats2.transpose()
print(train_stats2)

# Let's build our model. Here, we'll use a Sequential model with two densely connected hidden layers, and an output layer that returns a single, continuous value. 
print(":::::::::::::::::::: Here is the length of train_dataset.keys() :::::::::::::::::::::::::")
print(len(train_dataset.keys()))

train_dataset.keys()
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()

# Inspect the model
print(model.summary())

# Train the model for 1000 epochs, and record the training and validation accuracy in the history object.
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0,callbacks=[PrintDot()])

#Save the model
model.save('./')

# make a prediction with test data which weren't used in the training dataset.
inputs = normed_test_data[:10]
print('input data')
print(inputs)
test_predictions = model.predict(inputs).flatten()
print('predictions')
print(test_predictions)