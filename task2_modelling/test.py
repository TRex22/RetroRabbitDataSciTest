# My test file which is faster than using jupiter notebook

# imports
# Seaborn was only used for more complicated graphs
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# My own implementation
from GMM import GMM
from GMM import fn

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

# Training, test and validation sets
train_not_fraud, test_not_fraud = train_test_split(data_not_fraud, test_size=0.2, shuffle=True)
# test_not_fraud, validation_not_fraud = train_test_split(test_complete_not_fraud, test_size=0.1, shuffle=True)

train_fraud, test_fraud = train_test_split(data_fraud, test_size=0.2, shuffle=True)
# test_fraud, validation_fraud = train_test_split(test_complete_fraud, test_size=0.1, shuffle=True)

print("done setting up data.")

prior_train = len(train_fraud) / float(len(train_fraud) + len(train_not_fraud))
prior_test = len(test_fraud) / float(len(test_fraud) + len(test_not_fraud))

print("done setting up priors")

# 5. Train GMM
epsilon = 0.00001
train_count = 100
K = 2 #2 can be increased - train_fraud.shape[1]
print("K: ", K)

# Define GMMs
GMM_fraud = GMM(train_fraud, K)
GMM_not_fraud = GMM(test_not_fraud, 5)

# Train GMMs
GMM_fraud.train(max_count = train_count, epsilon = epsilon)
GMM_not_fraud.train(max_count = train_count, epsilon = epsilon)
# print(GMM_not_fraud.theta)
print("done training.")

cl = fn.classifier(test_fraud.values[0], GMM_fraud, GMM_not_fraud)
print("cl: ",cl)
bcl = fn.naive_bayes_classifier(test_fraud.values[0], GMM_fraud, GMM_not_fraud, prior_test)
print("bcl: ",bcl)



# Calculate training error
label_vector = np.ones(train_fraud.shape[0], dtype=bool)
train_fraud_err_classifier = fn.error(train_fraud, label_vector, GMM_fraud, GMM_not_fraud)
train_fraud_err_bayes_classifier = fn.bayes_error(train_fraud, label_vector, GMM_fraud, GMM_not_fraud, prior_train, confidence=0.65)
print("train_fraud_err (classifier): ", train_fraud_err_classifier)
print("train_fraud_err (bayes classifier): ", train_fraud_err_bayes_classifier)

label_vector = np.ones(train_not_fraud.shape[0], dtype=bool)
train_not_fraud_err_classifier = fn.error(train_not_fraud, label_vector, GMM_fraud, GMM_not_fraud)
train_not_fraud_err_bayes_classifier = fn.bayes_error(train_not_fraud, label_vector, GMM_fraud, GMM_not_fraud, prior_train, confidence=0.65)
print("train_not_fraud_err (classifier): ", train_not_fraud_err_classifier)
print("train_not_fraud_err (bayes classifier): ", train_not_fraud_err_bayes_classifier)

# 6. Error Measures
label_vector = np.ones(test_fraud.shape[0], dtype=bool)
test_fraud_err_classifier = fn.error(test_fraud, label_vector, GMM_fraud, GMM_not_fraud)
test_fraud_err_bayes_classifier = fn.bayes_error(test_fraud, label_vector, GMM_fraud, GMM_not_fraud, prior_test, confidence=0.65)
print("test_fraud_err (classifier): ", test_fraud_err_classifier)
print("test_fraud_err (bayes classifier): ", test_fraud_err_bayes_classifier)

label_vector = np.ones(test_not_fraud.shape[0], dtype=bool)
test_not_fraud_err_classifier = fn.error(test_not_fraud, label_vector, GMM_fraud, GMM_not_fraud)
test_not_fraud_err_bayes_classifier = fn.bayes_error(test_not_fraud, label_vector, GMM_fraud, GMM_not_fraud, prior_test, confidence=0.65)
print("test_not_fraud_err (classifier): ", test_not_fraud_err_classifier)
print("test_not_fraud_err (bayes classifier): ", test_not_fraud_err_bayes_classifier)
