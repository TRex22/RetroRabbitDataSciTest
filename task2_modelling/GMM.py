# Jason Chalom 2017
# GMM implementation I made for a computer vision course during my honours degree at Wits

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

def zero_init(training_data, K):
	lambda_vect = np.full((K), 1.0/K)

	# init randomly between (0,1] 
	# positive semi-def but already is
	# sigma_vect = np.full((K), np.var(training_data)) # diagonal

	sigma_list = []
	mean_list = []
	for k in range(K):
		mean = (1.-0.)*np.random.random_sample((training_data.shape[1])) + 0.
		mean_list.append(mean)

		sig = (1.0 -0.001)*np.random.random_sample((training_data.shape[1],training_data.shape[1])) + 0.001
		sig = np.dot(sig, sig.T)
		sig = np.diag(np.diag(sig))
		sigma_list.append(sig)

	sigma = np.array(sigma_list)
	mean_vect = np.array(mean_list)
	
	# print(mean_vect)
	# print(lambda_vect)

	return lambda_vect, mean_vect, sigma

def expectation_step(training_data, K, theta):
	# print (training_data)
	# print("Expectation Step")
	I = training_data.shape[0] #vector length
	l = np.zeros((I, K))
	r = np.zeros((I, K))

	lambda_vect = theta[0]
	mean_vect = theta[1]
	sigma_vect = theta[2]
	# print("Lambdas ", lambda_vect)
	# print("Mean ", mean_vect)
	# print("Sigmas ", sigma_vect)

	# Numerator of Bayes' rule
	for k in range(K):
		dist = multivariate_normal(mean=mean_vect[k], cov=sigma_vect[k])
		sample = dist.pdf(training_data)
		# print('sample: ', sample)
		l[:,k] = lambda_vect[k]*sample

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
	return r

def maximization_step(training_data, K, theta, r):
	# print("Maximization Step")
	I = training_data.shape[0] #vector length
	lambda_vect = theta[0]
	mean_vect = theta[1]
	sigma_vect = theta[2]

	sumri = np.sum(r, axis=0)
	# print "sumri", sumri
	# print "sumri sum", sumri.sum()
	lambda_vect = sumri/sumri.sum()
	
	for k in range(K):
		# optimize
		# r_sum = np.sum(r, axis=0)
		# r_k_sum = np.sum(r[:,k], axis=0)
		mean_vect[k] = r[:,k].dot(training_data) / sumri[k]

	for k in range(K):
		mean_shift = np.zeros(training_data.shape)
		mean_shift = np.subtract(training_data, mean_vect[k])
		sig = np.dot(mean_shift.T, np.multiply(r[:,k][:,np.newaxis], mean_shift))
		sigma_vect[k] = ((sig)) / (sumri[k])
		sigma_vect[k] = np.diag(np.diag(sigma_vect[k]))

	# print("Lambdas ", lambda_vect)
	# print("Mean ", mean_vect)
	# print("Sigmas ", sigma_vect)

	return lambda_vect, mean_vect, sigma_vect

def calc_probability(data, theta, K):	
	lambda_vect = np.copy(theta[0])
	mean_vect = np.copy(theta[1])
	sigma_vect = np.copy(theta[2])

	p = np.zeros(K)
	for k in range(K):
		sample = multivariate_normal.pdf(data, mean=mean_vect[k], cov=sigma_vect[k])
		# print(lambda_vect)
		p[k] = lambda_vect[k]*sample

	return p

	# out = 0.0		
	# for k in range(K):
	# 	out += lambda_vect[k]*(multivariate_normal.pdf(data, mean=mean_vect[k], cov=sigma_vect[k]))
	# return out

def calc_log_likelihood(training_data, theta, K):
	I = training_data.shape[0] #vector length
	lambda_vect = theta[0]
	mean_vect = theta[1]
	sigma_vect = theta[2]

	tol = 5000 # todo?
	loglikelihood = 0.0
	# for i in xrange(I):
	# 	inner = 0.0
	# 	for k in xrange(K):
	# 		dist = multivariate_normal(mean=mean_vect[k], cov=sigma_vect[k]).pdf(training_data[i])
	# 		inner = inner + (lambda_vect[k] * dist)

	# 	if inner != 0:
	# 		loglikelihood = loglikelihood + np.log(inner)

	inner_sum = 0.0
	for k in xrange(K):
		dist = multivariate_normal(mean=mean_vect[k], cov=sigma_vect[k])
		samples = dist.pdf(training_data)
		inner_sum = inner_sum + (lambda_vect[k]*samples)

	loglikelihood = np.sum(np.log(inner_sum), axis=0)

	print("loglikelihood: %f"%(loglikelihood))
	return loglikelihood

def label(prob, confidence):
	# hack something strange is happening with the probabilities they should be larger
	confidence = 0.00000000000000050
	sum = np.sum(prob)
	
	# print (sum)
	if (sum < confidence):
		return False;
	return True;

def error(test_vector, label_vector, theta, K, confidence=0.65):
	test_label = np.zeros(test_vector.shape[0])
	sum = 0
	for i in range(test_vector.shape[0]):
		p = calc_probability(test_vector.values[i], theta, K)
		# test_label[i] = np.argmax(p)#(p>confidence)
		test_label[i] = label(p, confidence)

		# print("%d : %d"%(test_label[i], label_vector[i]))
		if test_label[i] != label_vector[i]:
			# print(p)
		# if (p > confidence):
			sum = sum + 1

	# return np.sum(np.absolute(test_label-label_vector))/(label_vector.shape[0]*label_vector.shape[1])
	return sum/label_vector.shape[0]

def train(training_data, K, training_size=100, max_count = 255, epsilon = 0.001):
	count = 0
	
	theta = zero_init(training_data, K)
	
	prev_mean = np.zeros((K, training_data.shape[1]))
	mean = theta[1]
	mean_diff = np.linalg.norm(mean-prev_mean)
	
	while (count < max_count) and (mean_diff > epsilon): #epsilon and (L-L_old < epsilon)
		# print ("Iteration: ", count )
		prev_mean = np.copy(theta[1])

		r = expectation_step(training_data, K, theta)
		theta = maximization_step(training_data, K, theta, r)

		mean = theta[1]
		mean_diff = np.linalg.norm(mean-prev_mean)
		# print("Mean Diff: ",mean_diff)

		count = count + 1
		# print("\n\n")

		# Not using these:
		# L = calc_log_likelihood(training_data, K, theta)
		# Dont need EM bound?
		# B = calc_EM_bound(training_data, K, theta, r)

	return theta