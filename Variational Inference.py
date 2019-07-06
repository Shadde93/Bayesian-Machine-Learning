import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from scipy.special import gamma as Gamma
import math as mt


# Initial values
lambda_0 = 0
my_0 = 0
a_0 = 0
b_0 = 0

data = np.random.normal(loc = 0.0, scale = 1.0, size = 100)
numb = 100

my1 = np.linspace(-1, 1, numb)
my = my1[:,None]

tau1 = np.linspace(0,2, numb)
tau = tau1[:,None]


# parametres
N = len(data)
x_sum = sum(data)
x_bar = x_sum*1/N
x_mean_sq = sum(data**2)
my_N = (lambda_0 * my_0 + N * x_bar) / (lambda_0 + N)
a_N = a_0 + (N +1) / 2

def trueParam(N, lambda_0, my_N, x_sum, x_mean_sq, my_0, a_N):
    minIt = 1e-6
    lambda_N = 1
    b_0 = -1


    i = 1
    while(i < 100):

        b_N = b_0 + 0.5 * ((lambda_0 + N) * (lambda_N ** -1 + my_N ** 2) - 2 * (lambda_0 * my_0 + x_sum) * my_N + x_mean_sq + lambda_0 * (my_0 ** 2))

        lambda_N = (lambda_0 + N)*a_N/b_N


        if lambda_N - lambda_0 < minIt:
            return(lambda_N, b_N)

        lambda_0, b_0, i = lambda_N, b_N, i + 1
        #i = i + 1

lambda_true, b_true = trueParam(N, lambda_0, my_N, x_sum, x_mean_sq, my_0, a_N)



def truePosterior(lambda_0, b_0, a_0, my_0, tau, my, N, x_mean_sq, x_sum):

    a_star = N/2 + a_0
    b_star = 0.5*x_mean_sq+lambda_0*0.5*my_0**2 +b_0
    my_star = (x_sum+lambda_0*my_0)/(N+lambda_0)
    lambda_star = N+ lambda_0

    prior_tau = np.exp(-1*tau*b_star)*(np.power(tau, -1 + a_star))

    prior_my_tau = np.exp(np.dot((-1*(lambda_star*tau)/2) , np.transpose((my-my_star)**2)))


    posterior = prior_tau*prior_my_tau

    return (posterior)

Z1 = truePosterior(lambda_0, b_0, a_0, my_0, tau, my, N, x_mean_sq, x_sum)


def IVposterior(lambda_N, tau, my,my_N, b_N, a_N):


    q_my1 =  np.sqrt((lambda_N) / (2 * np.pi))* np.exp(np.dot((-1 * (lambda_N) / 2), np.transpose((my - my_N) ** 2)))

    q_tau1 =  (1 / mt.gamma(a_N)) * np.exp(-1 * tau * b_N) * (np.power(tau, -1 + a_N))

    IV = q_my1 * q_tau1

    return (IV)



a = IVposterior(lambda_true, tau, my, my_N, b_true, a_N)




X, Y = np.meshgrid(my1, tau1)
k = plt.contour(X, Y, Z1, colors = 'g')

#true posterior
l = plt.contour(X, Y, a, colors = 'b', linestyles = 'dashed')



plt.ylabel('tau')
plt.xlabel('mu')
plt.title('True posterior' )

plt.show()