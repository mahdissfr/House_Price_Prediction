# Import statements
import csv

import pandas
import sklearn
from sklearn.model_selection import train_test_split

from linear_regression import gd_train, load_data, preprocess_data, eval, plot_losses
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import savetxt

train_data = load_data("IA1_train.csv")
# train_data = load_data("data/PA1_train1.csv")
# train_data = train_data.sample(frac=1)  # shuffle
# x_val = load_data("data/PA1_test1.csv")

# #for testing with dev dataset:
val_data = load_data("data/IA1_dev.csv")
y_val = val_data['price']
x_val = val_data.drop(["price"], axis=1)



y_train = train_data['price']
x_train = train_data.drop(["price"], axis=1)
val_id = x_val['id']

# Preprocess training and validation data with Normalization
preprocessed_train_data, preprocessed_val_data = preprocess_data(x_train, x_val, 1, 0, 0)

weights, losses = gd_train(preprocessed_train_data, y_train, 0.0001)

predicted_price = np.dot(preprocessed_val_data, np.transpose(weights))


for x in sorted(zip(preprocessed_val_data.columns.values, abs(weights)), key=lambda x: x[1], reverse=True):
    print(x)
print("\n\n")
print("mse")
print(losses[-1])
print(eval(weights, preprocessed_val_data, y_val))

df_subm = pandas.DataFrame({'id': val_id, 'price': predicted_price})
df_subm.to_csv('out.csv', index=False)
