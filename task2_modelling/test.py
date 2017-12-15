# My test file which is faster than using jupiter notebook

# imports
# Seaborn was only used for more complicated graphs
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# My own implementation
import GMM as GMM

# 0. Load data (Need the csv file in the folder with the notebook)
data_file = "creditcard.csv"
df = pd.read_csv(data_file, delimiter=",", encoding="utf-8")

print("Shape of data:", df.values.shape)
print("Example Data:")
print(df.columns)
print(df.values[0])

# 1. clean up and normalise the data
# cleanup headings
df.columns = map(str.lower, df.columns)
df['class'] = df['class'].astype('bool')

# normalise data
# Used for boxplot
df_norm = (df.iloc[:,0:29]-df.iloc[:,0:29].mean())/df.iloc[:,0:29].std()

print("Data Import Done.")



# 4. Split data
data_not_fraud = df[df['class'] == False]
data_fraud = df[df['class'] == True]

include = ['v1', 'v2', 'v3', 'v4', 'v5', 'v7', 'v9', 'v10', 'v11', 'v12', 'v14', 'v16', 'v17', 'v18']
cols = [col for col in df.columns if col in include]

data_not_fraud = data_not_fraud[cols]
data_fraud = data_fraud[cols]

# # shuffle rows
# data_not_fraud = shuffle(data_not_fraud)
# data_fraud = shuffle(data_fraud)

# Training, test and validation sets
train_not_fraud, test_not_fraud = train_test_split(data_not_fraud, test_size=0.2, shuffle=True)
# test_not_fraud, validation_not_fraud = train_test_split(test_complete_not_fraud, test_size=0.1, shuffle=True)

train_fraud, test_fraud = train_test_split(data_fraud, test_size=0.2, shuffle=True)
# test_fraud, validation_fraud = train_test_split(test_complete_fraud, test_size=0.1, shuffle=True)

print("done setting up data.")




# 5. Train GMM

K = 2 #2 can be increased - train_fraud.shape[1]
print("K: ", K)
theta_fraud = GMM.train(train_fraud, K, training_size=train_fraud.shape[0], max_count = 1024, epsilon = 0.00001)
# theta_not_fraud = GMM.train(train_not_fraud, K, training_size=train_not_fraud.shape[0], max_count = 1024, epsilon = 0.00001)

# Calculate training error
label_vector = np.ones(train_fraud.shape[0], dtype=bool)
train_fraud_err = GMM.error(train_fraud, label_vector, theta_fraud, K)
print("train_fraud_err: ", train_fraud_err)

label_vector = np.zeros(train_not_fraud.shape[0], dtype=bool)
train_not_fraud_err = GMM.error(train_not_fraud, label_vector, theta_fraud, K)
print("train_not_fraud_err: ", train_not_fraud_err)

print("done training.")


# 6. Error Measures
label_vector = np.zeros(test_fraud.shape[0], dtype=bool)
test_fraud_err = GMM.error(test_fraud, label_vector, theta_fraud, K)
print("test_fraud_err: ", test_fraud_err)

label_vector = np.zeros(test_not_fraud.shape[0], dtype=bool)
test_not_fraud_err = GMM.error(test_not_fraud, label_vector, theta_fraud, K)
print("test_not_fraud_err: ", test_not_fraud_err)