import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import multivariate_normal
from scipy.spatial import distance


def generateDataset(): # 512 pos
    combinations = list(product([-1, 1], repeat=9))
    sets = []
    for l in combinations:
        arr = np.asarray(l)
        grid = np.reshape(arr,(3,3))
        sets.append(grid)
    return sets

def drawDataset(dataset):
    for i in range(3):
        print(" ",end="")
        for j in range(3):
            if dataset[i][j]== -1:
                print("X"," ",end="")
            else:
                print("O"," ",end="")
        print()


def model(number, dataset, theta):

    if number == 0:
        return 1/512
    p=1
    for i in range(3):# x^n_1 
        for j in range(3): # x^n_2 
            if number == 1:
                p = p * 1/(1+np.exp(-dataset[i,j]*theta[0]*(i-1)))
            if number == 2:
                p = p * 1/(1+np.exp(-dataset[i,j]*(theta[0]*(i-1) + theta[1]*(j-1))))
            if number == 3:
                p = p * 1/(1+np.exp(-dataset[i,j]*(theta[0]*(i-1) + theta[1]*(j-1)+theta[2])))
    return p

def priorSample(modelNumber, samples):
    sigma = 1000
    #A = np.random.rand(modelNumber, modelNumber)
    #A = sigma*np.ones((modelNumber, modelNumber))
    #B = np.dot(A, A.transpose())
    cov = sigma*np.eye(modelNumber)
    #cov = B
    #print(cov)
    #mean = np.zeros(modelNumber)
    mean = np.repeat(5, modelNumber)
    #print(mean)
    theta = np.random.multivariate_normal(mean, cov, samples)
    return theta

def computeEvidence(dataset, modelNumber, samples):
    p=0
    for i in range(len(samples)):
        p = p + model(modelNumber, dataset, samples[i])
    return p/len(samples)

def create_index_set(x):
    x = np.transpose(x)
    E = x.sum(axis=1)
    # change 'euclidean' to 'cityblock' for manhattan distance
    dist = distance.squareform(distance.pdist(x, 'euclidean'))

    np.fill_diagonal(dist, np.inf)
    
    L = []
    D = list(range(E.shape[0]))
    L.append(E.argmin())
    D.remove(L[-1])
    
    while len(D) > 0:
        # add d if dist from d to all other points in D
        # is larger than dist from d to L[-1]
        N = [d for d in D if dist[d, D].min() > dist[d, L[-1]]]
        
        if len(N) == 0:
            L.append(D[dist[L[-1],D].argmin()])
        else:
            L.append(N[dist[L[-1],N].argmax()])
        
        D.remove(L[-1])
        

    # reverse the resulting index array
    return np.array(L)[::-1]


samples1 = priorSample(1,5000)
samples2 = priorSample(2,5000)
samples3 = priorSample(3,5000)
l = generateDataset()

#print(len(samples1))
#print(len(samples2))
#print(len(samples3))


evidence = np.zeros([4,512])

for i in range(4):
    for j in range(512):
        if i == 0:
            evidence[i][j]=computeEvidence(l[j],i,samples1)
        if i == 1:
            evidence[i][j]=computeEvidence(l[j],i,samples1)
        if i == 2:
            evidence[i][j]=computeEvidence(l[j],i,samples2)
        if i == 3:
            evidence[i][j]=computeEvidence(l[j],i,samples3)

#print(evidence)
index = create_index_set(evidence)


max = np.argmax(evidence,axis=1)
min = np.argmin(evidence,axis=1)
sum = np.sum(evidence, axis=1)
#print(str(sum) + "the sums" )

#j = 0
#for i in max:
#    print(str(i)+ " = argmax for model " + str(j))
#    drawDataset(l[i])
#    j = j+1

#k = 0
#for q in min:
#    print(str(q) + " = argmin for model " + str(k))
#    drawDataset(l[q])
#    k = k + 1

#index = create_index_set(np.sum(evidence,axis=0))
#f1 = plt.figure()
#ax1 = f1.add_subplot(111)
#ax1.plot(evidence[0,index],'r', label = "p(D|M_0)", linewidth=1)
#ax1.plot(evidence[1,index],'g', label = "p(D|M_1)", linewidth=1)
#ax1.plot(evidence[2,index],'k', label = "p(D|M_2)", linewidth=1)
#ax1.plot(evidence[3,index],'b', label = "p(D|M_3)", linewidth=1)
#ax1 = plt.gca()
#ax1.set_xlabel('Dataset D')
#ax1.set_ylabel('Evidence')
#plt.legend()

f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.plot(evidence[0,index],'r', label = "p(D|M_0)", linewidth=1)
ax2.plot(evidence[1,index],'g', label = "p(D|M_1)", linewidth=1)
ax2.plot(evidence[2,index],'k', label = "p(D|M_2)", linewidth=1)
ax2.plot(evidence[3,index],'b', label = "p(D|M_3)", linewidth=1)
ax2 = plt.gca()
ax2.set_xlabel('Dataset D')
ax2.set_ylabel('Evidence')
ax2.set_xlim([0,100])
ax2.set_ylim([0,0.2])
plt.legend()

plt.show()
