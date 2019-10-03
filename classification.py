import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data')
X=y= data[:,:1],data[:,0]
data.head()


def sigmoid(X):
    sig= 1.0/(1.0 + e ** (-1.0 * X))
    return sig

def cost_fun(theta,X,y):
    m = X.shape[0]
    theta = reshape(theta,(len(theta),1))
    J = (1./m) * (-transpose(y).dot(log(sigmoid(X.dot(theta)))) - transpose(1-y).dot(log(1-sigmoid(X.dot(theta)))))

    grad = transpose((1./m)*transpose(sigmoid(X.dot(theta)) - y).dot(X))
    return J[0][0]

def gradient_fun(theta, X, y):
    grad = zeros(3)

    h = sigmoid(X.dot(theta.T))

    delta = h - y

    l = grad.size

    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * - 1

    theta.shape = (3,)

    return  grad

alpha = 0.1

def prediction(theta, X):
    m, n = X.shape[0]
    p = zeros(shape=(m, 1))
    h = sigmoid(X.dot(theta.T))

    for i in range(0, h.shape[0]):
        if h[i] > 0.5:
            p[i, 0] = 1
        else:
            p[i, 0] = 0

    return p

wr= np.array([2,20,400,95],dtype = np.int32)
wr= np.array([wr])
mr = np.prod(X.shape)
