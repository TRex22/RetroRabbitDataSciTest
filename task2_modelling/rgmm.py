#!/usr/bin/python
#Richard Klein 2017

import numpy as np
import numpy.random as rnd
from scipy.stats import multivariate_normal
import glob
import pickle
import pdb
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# fg = []
# bg = []

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
data_not_fraud = df_norm[df['class'] == False]
data_fraud = df_norm[df['class'] == True]

include = ['v1', 'v2', 'v3', 'v4', 'v5', 'v7', 'v9', 'v10', 'v11', 'v12', 'v14', 'v16', 'v17', 'v18']
# include = ['v1', 'v2', 'v3', 'v4', 'v5', 'v7', 'v9', 'v10', 'v11', 'v12', 'v14', 'v16', 'v17', 'v18', 'time', 'Amount']
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

fg = train_fraud
bg = train_not_fraud

print("done setting up data.")

fg = np.array(fg)
bg = np.array(bg)

prior = len(fg) / float(len(fg) + len(bg))
print(len(fg))
print(prior)

def random_cov(D):
	# Will ensure symmetric pos. def.
	out = rnd.random((D,D))
	out *= out.T
	out += 100*D*np.eye(D)
	# out += -0.001*D*np.eye(D)
	return out

class GMM():
	def __init__(self, data, K):
		self.data = data
		# Dimensionality
		self.D = len(data[0])
		# Data Size
		self.n = len(data)
		# Num Gaussians
		self.K = K
		# Init Uniform Lambdas
		self.lam = np.ones(K)/K
		# Init K Means [Dx1]
		self.mu = rnd.random((K,self.D))*1.
		# Init K cov matrices [DxD]
		self.cov = np.array([random_cov(self.D) for i in range(K)])
		# Init Responsibilities [n x K]
		self.r = np.zeros((self.n,self.K))
		
	def estep(self):
		# E-Step - Update Responsibility
		K = self.K
		n = self.n
		# Setup new dists
		self.setup()

		for i in range(n):
			sum = 0.0
			for k in range(K):
				self.r[i,k] = self.lam[k]*(self.norm[k].pdf(self.data[i]))
			self.r[i,:] /= self.r[i,:].sum()
			
	def mstep(self):
		# M-Step - Update Norm Params
		K = self.K
		D = self.D
		n = self.n
		sum_full = float(self.r.sum())
		r_sum = self.r.sum(0)		
		self.lam = r_sum/sum_full

		for k in range(K):
			self.mu[k] = self.r[:,k].dot(self.data)/r_sum[k]
		for k in range(K):
			tmp = np.zeros((D,D))
			for i in range(n):
				t = self.data[i,:] - self.mu[k]
				tmp += self.r[i,k]*np.outer(t,t)
			#print(tmp)
			self.cov[k] = tmp/r_sum[k]
			print("R_sum[k]: %f" % r_sum[k])
		
	def step(self):
		old_mu = self.mu.copy()
		print("E-Step")
		self.estep()
		print("M-Step")
		self.mstep()
		d = np.linalg.norm(old_mu - self.mu)
		print(d)
		return d

	def train(self, tol):
		d = tol
		while d >= tol:
			d = self.step()
			print("d: ",d)
		self.setup()

	def setup(self):
		K = self.K
		n = self.n
		self.norm = []
		for k in range(K):
			self.norm.append(multivariate_normal(mean=self.mu[k], cov=self.cov[k]))		

	def probs(self, x):
		K = self.K
		n = self.n
		
		out = 0.0		
		for k in range(K):
			out += self.lam[k]*(self.norm[k].pdf(x))
		return out	

b = GMM(bg, 3)
f = GMM(fg, 1)

print("BG")
b.train(20)
print(b)

print("FG")
f.train(20)

#For single pixels
def prob(x, b, f, prior):
	p1 = f.probs(x)
	p2 = b.probs(x)
	l1 = prior
	l2 = 1 - prior
	return (p1*l1)/(p1*l1 + p2*l2)

#For full images
def prob2(x, b, f, prior):
	p1 = f.probs(x)
	p2 = b.probs(x)
	l1 = prior
	l2 = 1 - prior
	return np.divide(p1*l1, p1*l1 + p2*l2)

p = prob2(fg[0], b, f, prior)
print(p)

#sss = {'f' : f, 'b' : b}
# #pickle.dump( sss, open( "save.pkl", "wb" ) )

# file_img = "./Training/t%d.jpg" % 1
# img = cv2.imread(file_img)
# p = prob2(img, b, f, prior)

def label(prob, confidence):
	# hack something strange is happening with the probabilities they should be larger
	# confidence = 0.00000000000000050
	# sum = np.sum(prob)
	
	# print (sum)
	if (prob >= confidence):
		return True;
	return False;

def error(test_vector, label_vector, theta, K, confidence=0.65):
	test_label = np.zeros(test_vector.shape[0])
	sum = 0
	for i in range(test_vector.shape[0]):
		# p = calc_probability(test_vector.values[i], theta, K)
		p = prob2(test_vector.values[i], b, f, prior)
		# test_label[i] = np.argmax(p)#(p>confidence)
		test_label[i] = label(p, confidence)

		# print("%d : %d"%(test_label[i], label_vector[i]))
		if test_label[i] != label_vector[i]:
			# print(p)
		# if (p > confidence):
			sum = sum + 1

	# return np.sum(np.absolute(test_label-label_vector))/(label_vector.shape[0]*label_vector.shape[1])
	return sum/label_vector.shape[0]



# Calculate training error
K=2

print("f model")
label_vector = np.ones(train_fraud.shape[0], dtype=bool)
train_fraud_err = error(train_fraud, label_vector, f, K)
print("train_fraud_err: ", train_fraud_err)

label_vector = np.zeros(train_not_fraud.shape[0], dtype=bool)
train_not_fraud_err = error(train_not_fraud, label_vector, f, K)
print("train_not_fraud_err: ", train_not_fraud_err)

label_vector = np.zeros(test_fraud.shape[0], dtype=bool)
test_fraud_err = error(test_fraud, label_vector, f, K)
print("\n\ntest_fraud_err: ", test_fraud_err)

label_vector = np.zeros(test_not_fraud.shape[0], dtype=bool)
test_not_fraud_err = error(test_not_fraud, label_vector, f, K)
print("test_not_fraud_err: ", test_not_fraud_err)



print("\n\nb model")
label_vector = np.ones(train_fraud.shape[0], dtype=bool)
train_fraud_err = error(train_fraud, label_vector, b, K)
print("train_fraud_err: ", train_fraud_err)

label_vector = np.zeros(train_not_fraud.shape[0], dtype=bool)
train_not_fraud_err = error(train_not_fraud, label_vector, b, K)
print("train_not_fraud_err: ", train_not_fraud_err)

label_vector = np.zeros(test_fraud.shape[0], dtype=bool)
test_fraud_err = error(test_fraud, label_vector, b, K)
print("\n\ntest_fraud_err: ", test_fraud_err)

label_vector = np.zeros(test_not_fraud.shape[0], dtype=bool)
test_not_fraud_err = error(test_not_fraud, label_vector, b, K)
print("test_not_fraud_err: ", test_not_fraud_err)