# Jason Chalom 2017
# GMM implementation I made for a computer vision course during my honours degree at Wits

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# These are functions which can be run on GMMs
class fn():
	def zero_init(data, K):
		lambda_vect = np.full((K), 1.0/K)

		# init randomly between (0,1] 
		# positive semi-def but already is
		# sigma_vect = np.full((K), np.var(data)) # diagonal

		sigma_list = []
		mean_list = []
		for k in range(K):
			mean = (1.-0.)*np.random.random_sample((data.shape[1])) + 0.
			mean_list.append(mean)

			sig = (1.0-0.001)*np.random.random_sample((data.shape[1],data.shape[1])) + 0.001
			sig = np.dot(sig, sig.T)
			sig = np.diag(np.diag(sig))
			sigma_list.append(sig)

		sigma = np.array(sigma_list)
		mean_vect = np.array(mean_list)

		# print(mean_vect)
		# print(lambda_vect)

		return lambda_vect, mean_vect, sigma

	def naive_bayes_classifier(data, GMM_fg, GMM_bg, prior, confidence=0.65):
		# test_label[i] = np.argmax(p)#(p>confidence)
		p1 = GMM_fg.probability(data)
		p2 = GMM_bg.probability(data)
		l1 = prior
		l2 = 1 - prior
		prob = np.divide(p1*l1, p1*l1 + p2*l2)

		# true if GMM_fg is greater
		if (prob > confidence):
			return True;
		return False;

	def classifier(data, GMM_fg, GMM_bg):
		# print("test")
		p1 = GMM_fg.probability(data)
		# print("test: ", p1)
		p2 = GMM_bg.probability(data)
		# print("test: ", p2)
		# true if GMM_fg is greater
		if (p1 > p2):
			return True;
		return False;

	def error(test_vector, label_vector, GMM_fg, GMM_bg):
		test_label = np.zeros(test_vector.shape[0])
		sum = 0
		for i in range(test_vector.shape[0]):
			test_label[i] = fn.classifier(test_vector.values[i], GMM_fg, GMM_bg)

			if test_label[i] != label_vector[i]:
				sum = sum + 1

		# return np.sum(np.absolute(test_label-label_vector))/(label_vector.shape[0]*label_vector.shape[1])
		return sum/label_vector.shape[0]

	def bayes_error(test_vector, label_vector, GMM_fg, GMM_bg, prior, confidence=0.65):
		test_label = np.zeros(test_vector.shape[0])
		sum = 0
		for i in range(test_vector.shape[0]):
			test_label[i] = fn.naive_bayes_classifier(test_vector.values[i], GMM_fg, GMM_bg, prior, confidence)

			if test_label[i] != label_vector[i]:
				sum = sum + 1

		# return np.sum(np.absolute(test_label-label_vector))/(label_vector.shape[0]*label_vector.shape[1])
		return sum/label_vector.shape[0]

class GMM():
	def __init__(self, data, K):
		self.data = data
		# Dimensionality
		# self.D = len(data[0])
		# Data Size
		self.I = data.shape[0]
		# Num Gaussians
		self.K = K

		self.theta = fn.zero_init(self.data, self.K)
		
		# Init Responsibilities [n x K]
		self.r = np.zeros((self.I,self.K))

	def expectation_step(self):
		# print("Expectation Step")
		I = self.I #vector length
		K = self.K

		l = np.zeros((I, K))
		r = np.zeros((I, K))

		lambda_vect = self.theta[0]
		mean_vect = self.theta[1]
		sigma_vect = self.theta[2]
		# print("Lambdas ", lambda_vect)
		# print("Mean ", mean_vect)
		# print("Sigmas ", sigma_vect)

		# Numerator of Bayes' rule
		for k in range(K):
			dist = multivariate_normal(mean=mean_vect[k], cov=sigma_vect[k])
			sample = dist.pdf(self.data)
			# print('sample: ', sample)
			l[:,k] = lambda_vect[k]*sample

		# Compute posterior by normalizing ...
		l_k_sum = np.sum(l, axis=1)

		# another hack to deal with singularities
		if(l_k_sum.any() == 0.):
			print("l_k_sum is 0")

		# Compute posterior by normalizing ...
		l_k_sum = np.sum(l, axis=1)

		for i in range(I):
			# r[:][k] = 1.00*l[:][k] / 1.00*l_i_sum
			# print "numerator: ",l[:,k]
			# print "lisum[k]: ", l_i_sum[k]
			# print "r: ", l[:,k]/l_i_sum[k]
			r[i,:] = l[i,:]/l_k_sum[i]

		# print("r: ", r)
		# print("r shape: ", r.shape)
		# print("r_sum: ", np.sum(r,axis=0))
		self.r = r

	def maximization_step(self):
		# print("Maximization Step")

		I = self.I #vector length
		K = self.K

		lambda_vect = self.theta[0]
		mean_vect = self.theta[1]
		sigma_vect = self.theta[2]

		sumri = np.sum(self.r, axis=0)
		# print("sumri",  self.r)
		# print "sumri sum", sumri.sum()
		lambda_vect = sumri/sumri.sum()
		
		for k in range(K):
			# optimize
			# r_sum = np.sum(r, axis=0)
			# r_k_sum = np.sum(r[:,k], axis=0)
			mean_vect[k] = self.r[:,k].dot(self.data) / sumri[k]

		for k in range(K):
			mean_shift = np.zeros(self.data.shape)
			mean_shift = np.subtract(self.data, mean_vect[k])
			sig = np.dot(mean_shift.T, np.multiply(self.r[:,k][:,np.newaxis], mean_shift))
			sigma_vect[k] = ((sig)) / (sumri[k])
			sigma_vect[k] = np.diag(np.diag(sigma_vect[k]))

		# print("Lambdas ", lambda_vect)
		# print("Mean ", mean_vect)
		# print("Sigmas ", sigma_vect)

		self.theta = lambda_vect, mean_vect, sigma_vect

	def probability(self, data):	
		lambda_vect = np.copy(self.theta[0])
		mean_vect = np.copy(self.theta[1])
		sigma_vect = np.copy(self.theta[2])

		# p = np.zeros(K)
		p = 0.0
		for k in range(self.K):
			sample = multivariate_normal.pdf(data, mean=mean_vect[k], cov=sigma_vect[k])
			# print(lambda_vect)
			p = p + (lambda_vect[k]*sample)

		return p

	def calc_log_likelihood(self):
		I = self.I #vector length
		lambda_vect = self.theta[0]
		mean_vect = self.theta[1]
		sigma_vect = self.theta[2]

		tol = 5000 # todo?
		loglikelihood = 0.0
		# for i in xrange(I):
		# 	inner = 0.0
		# 	for k in xrange(K):
		# 		dist = multivariate_normal(mean=mean_vect[k], cov=sigma_vect[k]).pdf(data[i])
		# 		inner = inner + (lambda_vect[k] * dist)

		# 	if inner != 0:
		# 		loglikelihood = loglikelihood + np.log(inner)

		inner_sum = 0.0
		for k in range(self.K):
			dist = multivariate_normal(mean=mean_vect[k], cov=sigma_vect[k])
			samples = dist.pdf(self.data)
			inner_sum = inner_sum + (lambda_vect[k]*samples)

		loglikelihood = np.sum(np.log(inner_sum), axis=0)

		print("loglikelihood: %f"%(loglikelihood))
		return loglikelihood

	def train(self, max_count = 255, epsilon = 0.001):
		count = 0
		prev_mean = np.zeros((self.K, self.data.shape[1]))
		mean = self.theta[1]
		mean_diff = np.linalg.norm(mean-prev_mean)
		
		while (count < max_count) and (mean_diff > epsilon): #epsilon and (L-L_old < epsilon)
			# print ("Iteration: ", count )
			prev_mean = np.copy(self.theta[1])

			# The easiest fix for singualr covd
			if(np.isnan(self.theta[0]).any() or np.isnan(self.theta[1]).any() or np.isnan(self.theta[2]).any()):
				self.theta = fn.zero_init(self.data, self.K)

			self.expectation_step()
			self.maximization_step()

			if(np.isnan(self.theta[0]).any() or np.isnan(self.theta[1]).any() or np.isnan(self.theta[2]).any()):
				self.theta = fn.zero_init(self.data, self.K)

			mean = self.theta[1]
			mean_diff = np.linalg.norm(mean-prev_mean)
			# print("Mean Diff: ",mean_diff)

			count = count + 1
			# print("\n\n")

			# Not using these:
			# L = calc_log_likelihood(self.data, self.K, self.theta)
			# Dont need EM bound?
			# B = calc_EM_bound(self.data, self.K, self.theta, self.r)
		# print(self.theta)
		return self.theta