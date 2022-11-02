from statistics import mean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy
from matplotlib import pyplot
from numpy import std
from scipy.stats import pearsonr


def check_correlation(data1, data2):
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: %.3f' % corr)
    # plot
    pyplot.scatter(data1, data2)
    pyplot.show()


def load_data(path):
    """
    load_data loads in the csv at the provided path
    :param path: the path to the csv file
    :return data: a dataframe of the csv file 
    """

    data = pd.read_csv(path)
    return data


def feature_eng(data):
    # condition = (data['yr_renovated'] == 0)
    # data.loc[condition, 'yr_renovated'] = data.loc[condition, 'yr_built']
    #
    # print('\nyr_renovated' + '              age_since_renovated')
    # check_correlation(data['yr_renovated'], data['age_since_renovated'])


    data['total_sqft'] = (data['sqft_living'] + data['sqft_above'] + data['sqft_basement'])**3
    data['view_waterfront'] = 7 * data['waterfront'] + data['view']

    #right skewed
    data["lat"] = np.log(data["lat"])
    data["year"] = np.log(data["year"])
    #left skewed
    data["bathrooms"] = data["bathrooms"] ** 3
    data["view"] = data["view"] ** 3
    data["condition"] = data["condition"] ** 3
    data["grade"] = data["grade"] ** 3
    data["long"] = data["long"] ** 3

    data = data.drop(["sqft_living"], axis=1)
    data = data.drop(["sqft_above"], axis=1)
    data = data.drop(["sqft_basement"], axis=1)

    data = data.drop(["bedrooms"], axis=1)
    data = data.drop(["day"], axis=1)
    data = data.drop(["sqft_lot"], axis=1)
    data = data.drop(["yr_built"], axis=1)
    data = data.drop(["sqft_lot15"], axis=1)
    data = data.drop(["month"], axis=1)

    return data


def preprocess_helper(data, drop_sqft_living15, solo):
    """
    preprocess_helper This function preprocess the provded data, but does not normalize it and is a helper funtion for 
    the preprocess_data function
    :param data: the data to be preprocessed
    :param drop_sqft_living15: a boolean that indicates if sqft_living should be dropped
    :return data: a dataframe of the preprocessed data
    """

    # part 0 (a)
    data = data.drop(['id'], axis=1)

    # part 0 (b)
    month = []
    day = []
    year = []

    for line in data['date']:
        date = line.split('/')
        month.append(int(date[0]))
        day.append(int(date[1]))
        year.append(int(date[2]))

    data['month'] = month
    data['day'] = day
    data['year'] = year
    data = data.drop(['date'], axis=1)

    # part 0 (c)
    data.insert(loc=0, column='dummy', value=np.ones(len(data)))

    # part 0 (d) 
    # row index 10 has a value of -1
    age_since_renovated = []
    for index, row in data.iterrows():
        if row["yr_renovated"] == 0:
            age_since_renovated.append(int(row["year"]) - row["yr_built"])
        else:
            age_since_renovated.append(int(row["year"]) - row["yr_renovated"])
    data["age_since_renovated"] = age_since_renovated

    # part 2 (b)
    if drop_sqft_living15:
        data = data.drop(['sqft_living15'], axis=1)

    if solo:
        data = feature_eng(data)

    return data


def preprocess_data(train_data, val_data, normalize, drop_sqft_living15, solo):
    """
    preprocess_data This function preprocess the provded data
    :param train_data: the training data to be preprocessed
    :param val_data: the validation data to be preprocessed
    :param normalize: a boolean idicating wether or not to normalize the data
    :param drop_sqft_living15: a boolean that indicates if sqft_living should be dropped
    :return train_data: a dataframe of the preprocessed training data
    :return val_data: a datafram of the preprocessed validation data
    """

    train_data = preprocess_helper(train_data, drop_sqft_living15, solo)
    val_data = preprocess_helper(val_data, drop_sqft_living15, solo)

    # part 0 (e)
    if normalize:
        for col in train_data:
            if col not in ["waterfront", "price", "dummy"]:
                print(col)
                mean = np.mean(train_data[col])
                std = np.std(train_data[col])
                # print(len(train_data[col]))
                # print(train_data[col][0])

                train_data[col] = (train_data[col].astype(float) - mean) / std
                val_data[col] = (val_data[col].astype(float) - mean) / std
                # val_data[col] = [float((val_data[col][idx]) - mean) / std for idx in range(0, len(val_data[col]))]
    return train_data, val_data


# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return modified_data


def converge(losses):
    """
    converge This tests the losses to see of the model has converged
    :param losses: a list of the model's losses
    :return: a boolean that indicates convergence
    """

    if (len(losses)) > 2:
        mse = losses[-1]
        prev_mse = losses[-2]
        if mse < 0.001:
            return True
        if (prev_mse - mse) < 0.001:
            return True
    return False


def diverge(losses):
    """
    diverge This tests the losses to see of the model has diverged
    :param losses: a list of the model's losses
    :return: a boolean that indicates divergence
    """

    if (len(losses)) > 2:
        mse = losses[-1]
        prev_mse = losses[-2]
        if prev_mse < mse:
            return True
    return False


# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(features, y, lr):
    """
    gd_train This function trains a linear regression model using batch gradient descent using the provided learning rate
    :param features: the feature data to train with
    :param y: the actual price values
    :param lr: the learning rate to use for training
    :return weights: the weights of the trained model
    :return losses: a list of the losses for each training iteration
    """

    # # Select random weight values
    weights = np.random.rand(features.shape[1])

    # Initialize losses array
    losses = []

    MSE = 10000
    iterations = 0

    # while iterations < (4000) and not converge(losses) and not diverge(losses):
    while iterations < 4000 and not converge(losses):
        predicted_price = np.dot(features, np.transpose(weights))
        # if iterations % 1000 == 0:
        #     print(iterations, MSE)
        # loss function
        diff = predicted_price - y
        MSE = (sum((diff) ** 2)) / len(y)
        losses.append(MSE)

        # Gradient Descent for MSE
        gradient = (2 * np.dot(diff, features)) / len(features)

        weights = weights - (lr * gradient)  # 𝐰 ← 𝐰 − 𝛾 (𝛻𝐿(𝐰))
        # print(np.linalg.norm(gradient))

        iterations += 1

    return weights, losses


def eval(weights, features, y):
    """
    eval This tests returns the MSE for proved weights on a given set of data
    :param weights: the weights to use for evaluation
    :param features: the features that we are using to predict the housing price
    :param y: the actual prices for the given data 
    :return MSE: the mean squared area for the features using the provided weights 
    """
    predicted_price = np.dot(features, np.transpose(weights))
    # loss function
    diff = predicted_price - y
    MSE = (sum((diff) ** 2)) / len(y)

    return MSE


# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses, labels, filepath):
    """
    plot_losses This function plots the losses
    :param losses: A list of the losses to be plotted
    :param labels: A list of the labels for each plot
    :param filepath: the location where the file should be saved
     """
    plt.xlabel('No. of iterations')
    plt.ylabel('Losses')
    plt.title('Loss Function Plot')
    for i in range(0, len(losses)):
        plt.plot(losses[i], label=labels[i])
    plt.legend()
    plt.savefig(filepath)
