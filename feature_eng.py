# generate related variables
import sns as sns
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
# sudo apt-get install python3-tk
from linear_regression import *
from scipy.stats import pearsonr, spearmanr


# seed random number generator

def check_correlation(data1, data2):
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: %.3f' % corr)
    corr, _ = spearmanr(data1, data2)
    print('Spearmans correlation: %.3f' % corr)
    # plot
    pyplot.scatter(data1, data2)
    pyplot.show()


def find_outliers_IQR(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]
    outliers2 = df[df > (q3 + 1.5 * IQR)]

    print(len(outliers))
    print(len(outliers2))
    print("11111")
    return outliers2


# Load in data using relative file path
train_data = load_data("data/PA1_train1.csv")
train_data = train_data.sample(frac=1)  # shuffle
val_data = load_data("data/PA1_test1.csv")
val_id = val_data['id']
train_data['total_sqft'] = train_data['sqft_living'] + train_data['sqft_above'] + train_data['sqft_basement']
train_data['view_waterfront'] = 7*train_data['waterfront'] + train_data['view']
dsc = train_data.describe()
# print(dsc.floors)
# print(len(train_data["waterfront"]))
# print(len(train_data["view"]))
# out1 = find_outliers_IQR(train_data["floors"])
# train_data = train_data.drop(out1,axis=0)
# Preprocess training and validation data with Normalization

# condition  = (train_data['yr_renovated'] == 0 )
# train_data.loc[condition, 'yr_renovated'] = train_data.loc[condition, 'yr_built']
#

# for x in zip(train_data['total_sqft2'], train_data['total_sqft'], train_data['sqft_lot']):
#     print(x)
# check_correlation(train_data['yr_renovated'], train_data['yr_built'])
#
# #


# condition = (train_data['yr_renovated'] == 0)
# train_data.loc[condition, 'yr_renovated'] = train_data.loc[condition, 'yr_built']

# train_data = train_data.drop(["yr_renovated"], axis=1)

# print("\n\n\n")
preprocessed_train_data, preprocessed_val_data = preprocess_data(train_data, val_data, 0, 0)

print(abs(preprocessed_train_data.corr()['price']).sort_values())
# print("\n\n")
variance = sorted(zip(preprocessed_train_data.var(), train_data.columns.values), key=lambda x: x[0], reverse=True)
for x in variance:
    print(x)

# print('\nyr_renovated'+'              age_since_renovated')
# check_correlation(preprocessed_train_data['yr_renovated'], preprocessed_train_data['age_since_renovated'])

# print('\nsqft_lot15'+'              sqft_lot')
# check_correlation(preprocessed_train_data['sqft_lot15'], preprocessed_train_data['sqft_lot'])
# print('\nsqft_living15'+'              sqft_living')
# check_correlation(preprocessed_train_data['sqft_living15'], preprocessed_train_data['sqft_living'])
# print('\ntotal_sqft'+'              sqft_living')
# check_correlation(preprocessed_train_data['total_sqft'], preprocessed_train_data['sqft_living'])
# print('\ntotal_sqft'+'              sqft_above')
# check_correlation(preprocessed_train_data['total_sqft'], preprocessed_train_data['sqft_above'])
# print('\ntotal_sqft'+'              sqft_basement')
# check_correlation(preprocessed_train_data['total_sqft'], preprocessed_train_data['sqft_basement'])
# print('\nview_waterfront'+'              waterfront')
# check_correlation(train_data['view_waterfront'], preprocessed_train_data['waterfront'])

print('\nyr_renovated'+'              age_since_renovated')
check_correlation(preprocessed_train_data['yr_renovated'], preprocessed_train_data['age_since_renovated'])

#


#
# Weight values for our model trained on normalized data
# ('dummy', 5.31397452004911, 5.31397452004911)
# ('waterfront', 1.23360143823435, 1.23360143823435)
# ('sqft_basement', 0.2609555134927365, 0.2609555134927365)
# ('month', 0.2609006960887462, 0.2609006960887462)
# ('yr_built', 0.25117722674201376, 0.25117722674201376)
# ('year', 0.24903018124677065, 0.24903018124677065)
# ('sqft_living', 0.24258588173696322, -0.24258588173696322)
# ('age_since_renovated', 0.22541266625616502, 0.22541266625616502)
# ('sqft_above', 0.17369683856711945, 0.17369683856711945)
# ('long', 0.12417317802424967, 0.12417317802424967)
# ('sqft_lot', 0.0902581483458586, 0.0902581483458586)
# ('zipcode', 0.07500874786012823, 0.07500874786012823)
# ('day', 0.0711016637507016, 0.0711016637507016)
# ('floors', 0.06028948256801905, 0.06028948256801905)
# ('view', 0.052330943068784525, -0.052330943068784525)
# ('grade', 0.03189492215237474, -0.03189492215237474)
# ('bedrooms', 0.02496328908310579, -0.02496328908310579)
# ('condition', 0.013185452999126232, 0.013185452999126232)
# ('lat', 0.01010522724710456, -0.01010522724710456)
# ('bathrooms', 0.000196476303432135, -0.000196476303432135)
