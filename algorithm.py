##################################################
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import sys
import random
from helper import *
np.random.seed(0)

#########################Least Square Rregression using SGD##########################

# alpha is the regularization coeffecient, b the mini-batch size and step is the learning rate
def LSR(x, y, theta_init, maxsteps, b, step, random_state, alpha):
    # A random state "rng" has to be passed into the 
    # function so as to ensure that while 
    # parallelizing the code no randomness is 
    # additionally added because of seperate processes 
    # and the experiments are deterministic.
    rng = np.random.RandomState(random_state) 
    m = y.size # number of data points
    neworder = rng.permutation(m) # permutation of the data
    x = x[neworder] #features
    y = y[neworder] #labels
    # an array "history" that stores all the parameters. I store everything 
    # because in our final plots we might need them if 
    # we make some non-standard plots 
    history = []  
    theta = theta_init 
    #it counts the number of iterations. 
    #In our theory it is indexed by k for example.
    counter = 0 
    i = 0 #counts total gradients computed until now
    gradient = 0 
    for j in range(b):
        pred = np.dot(x[i+j], theta)
        error = pred - y[i+j]
        gradient = gradient + x[i+j].T*error #standard SGD teps for LSR 
    gradient = gradient/float(b) + alpha*theta #accounting for the mini-batch size
    while 1:
        theta = theta - step * gradient  # update        
        history.append(theta)
        counter+=1
        i += b #Remember, using a batch size of b
        
        if i >= m-b:# reached past the end.
            neworder = rng.permutation(m)
            x = x[neworder]
            y = y[neworder]
            i = 0 #reshuffling the data to keep everything random
        
        if counter == maxsteps:
            break

        gradient = 0 
        for j in range(b): #recomputing
            pred = np.dot(x[i+j], theta)
            error = pred - y[i+j]
            gradient = gradient + x[i+j].T*error 
        gradient = gradient/float(b) + alpha*theta

    return history #return all the parameters for the run

######################Logistic Regression############################################

# "alpha" is the regularization coeffecient, "b" the mini-batch size, "step" is the passed 
# as the learning rate constant, "decay" specifies the type of learning rate (constant or decaying). "step" 
# is passed so that local_SGD knows where to start from. If decay=0, then "step" functions 
# simply as a constant learning rate, else "step_init" specifies how many steps have already
# been taken on the machine and what annealing is to be used for the constant "step".   
def LOG(x, y, theta_init, maxsteps, b, step, step_init, decay, alpha, random_state ):
    # A random state "rng" has to be passed into the 
    # function so as to ensure that while 
    # parallelizing the code no randomness is 
    # additionally added because of seperate processes 
    # and the experiments are deterministic.
    rng = np.random.RandomState(random_state) 
    m = y.size # number of data points
    neworder = rng.permutation(m) # permutation of the data
    x = x[neworder] #features
    y = y[neworder] #labels
    # an array "history" that stores all the parameters. I store everything 
    # because in our final plots we might need them if 
    # we make some non-standard plots 
    history = []  
    theta = theta_init 
    #it counts the number of iterations. 
    #In our theory it is indexed by k for example.
    if decay == 0:
        counter = 0 
        i = 0 #counts total gradients computed until now
        gradient = 0 
        for j in range(b):
            z = np.dot(x[j],theta)
            prob = sigmoid(z) #this is sigmoid
            gradient = gradient + x[j]*(prob-y[j]) #this is gradient. standard SGD teps for LOG 
        gradient = gradient/float(b) + alpha*theta #accounting for the mini-batch size
        while 1:
            theta = theta - step * gradient  # update        
            history.append(theta)
            counter+=1
            i += b #Remember, using a batch size of b
            if i >= m-b:# reached past the end.
                neworder = rng.permutation(m)
                x = x[neworder]
                y = y[neworder]
                i = 0 #reshuffling teh data to keep everything random
            if counter == maxsteps:
                break
            gradient = 0 
            for j in range(b): #recomputing
                z = np.dot(x[j],theta)
                prob = sigmoid(z) #this is sigmoid
                gradient = gradient + x[j]*(prob-y[j]) #this is gradient. standard SGD teps for LOG 
            gradient = gradient/float(b) + alpha*theta
        return history #return all the parameters for the run
    else:
        counter = 0 
        i = 0 #counts total gradients computed until now
        gradient = 0 
        for j in range(b):
            z = np.dot(x[j],theta)
            prob = sigmoid(z) #this is sigmoid
            gradient = gradient + x[j]*(prob-y[j]) #this is gradient. standard SGD teps for LOG 
        gradient = gradient/float(b) + alpha*theta #accounting for the mini-batch size
        while 1:
            theta = theta - step/np.sqrt((step_init+counter+1)) * gradient  # update        
            history.append(theta)
            counter+=1
            i += b #Remember, using a batch size of b
            if i >= m-b:# reached past the end.
                neworder = rng.permutation(m)
                x = x[neworder]
                y = y[neworder]
                i = 0 #reshuffling teh data to keep everything random
            if counter == maxsteps:
                break
            gradient = 0 
            for j in range(b): #recomputing
                z = np.dot(x[j],theta)
                prob = sigmoid(z) #this is sigmoid
                gradient = gradient + x[j]*(prob-y[j]) #this is gradient. standard SGD teps for LOG 
            gradient = gradient/float(b) + alpha*theta
        return history #return all the parameters for the run

