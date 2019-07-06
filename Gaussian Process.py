import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist, pdist

# Generate data
l = 1.4
sigma = 1.0
cardinality = 7
noise_sigma = 0.5

X = np.linspace(0, 2*np.pi, num=cardinality)
Xlen = X.shape[0]
Y = np.cos(X) + np.sqrt(noise_sigma)*np.random.randn(Xlen)
x_s = np.linspace(-3, 9, 500) #x_star with 500 points
x_s_len = x_s.shape[0]
X = X[:,None]
Y = Y[:,None]
x_s = x_s[:,None]

def kernel(xi, xj, scale):
	return (sigma**2)*np.exp(-cdist(xi, xj, 'sqeuclidean')/(scale**2)) #eq 17

def compute_posterior(scale):
    k = kernel(x_s,X,scale)
    Cinv = np.linalg.inv(kernel(X,X,scale))
    mu = np.dot(np.dot(k,Cinv),Y)

    c = kernel(x_s, x_s,scale)
    sig = c - np.dot(np.dot(k,Cinv),np.transpose(k))
    return mu, sig

def plot_sample_prior(scale):
	samples = 10
	mu = np.zeros((x_s_len))
	K =  kernel(x_s,x_s,scale)
	Z = np.random.multivariate_normal(mu,K,samples)
	plt.figure()
	for i in range(samples):
		plt.plot(x_s[:],Z[i,:])
	title = 'length-scale '+str(scale)
	plt.title(title)
	plt.show()

def plot_samples_posterior(mu, sig):
	samples = 10
	mu = np.reshape(mu,(x_s_len,))
	Z = np.random.multivariate_normal(mu,np.nan_to_num(sig),samples)
	plt.figure()
	plt.plot(X,Y,'ko')
	for i in range(samples):
		plt.plot(x_s[:],Z[i,:])
	#plt.plot(x_s[:],mu[:],color="black",linewidth=2)
	#title = 'length-scale '+str(l)
	#plt.title(title)
	plt.show()

def plot_posterior(mu, sig):
    plt.plot(X, Y,'ko')
    plt.plot(x_s,np.cos(x_s), color = 'blue')
    plt.plot(x_s,mu, color = 'red')
    mu = np.reshape(mu,(x_s_len,))
    xs = np.reshape(x_s,(x_s_len,))
    sig_vec = np.diag(sig)
    upper = mu + 1*sig_vec
    lower = mu - 1*sig_vec
    ax = plt.gca()
    ax.fill_between(xs, upper, lower, facecolor='red', interpolate=True, alpha=0.09)
    #title = 'l= '+str(l)
    #plt.title(title)
    plt.show()

mu, sig = compute_posterior(l)

plot_sample_prior(l)

#plot_samples_posterior(mu,sig)
#plot_posterior(mu, sig)
