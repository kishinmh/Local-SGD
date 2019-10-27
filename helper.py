##################################################
import numpy as np
from numpy.random import randn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import Parallel, delayed
import sys
import os
import random
np.random.seed(0)

##################Helper Functions for algorithms and main routine##################
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def accuracy(x,y,w):
    z = np.dot(x,w)
    prob = sigmoid(z)
    pred = np.round(prob)
    return accuracy_score(y,pred)


#gives the normalized objective error
def nl(train_x, train_y, optimal, pred):
    optimal_loss = mean_squared_error(np.dot(train_x,optimal),train_y)
    curr_loss = mean_squared_error(pred,train_y)
    return (curr_loss-optimal_loss)/optimal_loss
    #return curr_loss-optimal_loss

# Returns the loss given the history of iterates. If "tail" is not zero gives 
# the tail averaging of last "tail" samples instead of just averaging.
def error_maker(history, optimal, train_x, train_y, sampling, tail):
    pred_in_time = np.matmul(history[0::sampling], np.transpose(train_x)) # predictions made by the current iterate
    #print "Heavy Matrix Operation Done!"

    loss = []
    for item in pred_in_time:
        loss.append(nl(train_x,train_y,optimal,item)) # loss for the current iterate
    
    #Finding the average or tail averaged estimators over time.
    average = []
    temp=0
    i=0
    if tail==0:
        for item in history:
            val = (i*temp + item)/(i+1)
            average.append(val)
            temp = val
            i=i+1
    else:
        for item in history:
            if i<tail:
                average.append(item)
                i=i+1
            else:
                average.append(np.mean(history[i-tail:i],0))
                i=i+1

    average_pred_in_time = np.matmul(average[0::sampling],np.transpose(train_x)) # predictions made by the average iterate
    #print "Heavy Matrix Operation Done!"
    
    average_loss=[]
    for item in average_pred_in_time:
        average_loss.append(nl(train_x,train_y,optimal,item)) # loss of the average iterate
    
    return loss, average_loss

def error_maker_log(history, train_x, train_y, sampling, tail):
    pred_in_time = np.round(sigmoid(np.matmul(history[0::sampling], np.transpose(train_x)))) # predictions made by the current iterate
    #print "Heavy Matrix Operation Done!"

    loss = []
    for item in pred_in_time:
        loss.append(1-accuracy_score(train_y,item)) # loss for the current iterate
    
    #Finding the average or tail averaged estimators over time.
    average = []
    temp=0
    i=0
    if tail==0:
        for item in history:
            val = (i*temp + item)/(i+1)
            average.append(val)
            temp = val
            i=i+1
    else:
        for item in history:
            if i<tail:
                average.append(item)
                i=i+1
            else:
                average.append(np.mean(history[i-tail:i],0))
                i=i+1

    average_pred_in_time = np.round(sigmoid(np.matmul(average[0::sampling],np.transpose(train_x)))) # predictions made by the average iterate
    #print "Heavy Matrix Operation Done!"
    
    average_loss=[]
    for item in average_pred_in_time:
        average_loss.append(1-accuracy_score(train_y,item)) # loss of the average iterate
    
    return loss, average_loss

def data_loader(name):
    x = np.load("Dataset/"+str(name)+"/train_x.npy")
    y = np.load("Dataset/"+str(name)+"/train_y.npy")
    d=len(x[0])
    if name not in ["Splice","Gisette", "Default", "Covtype", "Higgs", "ijcnn1", "Madelon"]:
        optimal = np.array(np.linalg.lstsq(x,y,rcond=None)[0])
    else:
        path = "Dataset/"+str(name)+"/optimal.npy"
        if os.path.isfile(path):
            optimal = np.load(path)
        else:
            optimal = np.zeros(len(x[0])) 
    return x, y, optimal

#Generating synthetic dataset
def data_maker(n,d,eigen_decay,cova_noise,noise):
    #The optimal point \ww^\star with dimension d, sampled normally
    optimal = randn(d)
    #A random d*d matrix
    random_matrix = randn(d,d)
    #Obtaining unitary matrices from this matrix
    #s is a diagonal matrix we'd replace further
    u,s,v = np.linalg.svd(random_matrix)
    if eigen_decay == 1: #eigenvalues increase as 1/p
        cova = cova_noise*np.dot(u,np.dot(np.diag(np.sqrt(np.arange(d)+1)),v))
    elif eigen_decay == 2: #eigenvalues are constant
        cova = cova_noise*np.dot(u,np.dot(np.eye(d),v))
    elif eigen_decay == 3: #eigenvalues decay as p
        cova = cova_noise*np.dot(u,np.dot(np.diag(1/np.sqrt(np.arange(d)+1)),v))
    #normalizing the optimal point with cova
    optimal = (optimal/np.linalg.norm(np.dot(optimal,cova))).reshape(d,1)
    #The features are simply normally sampled with covariance cova
    train_x = np.dot(randn(n,d),cova)
    #The prediction is optimal time feature plus some noise  
    train_y = np.dot(train_x,optimal).reshape(n,1) + noise*randn(n,1)
    optimal = np.array(np.linalg.lstsq(train_x,train_y,rcond=None)[0])
    #also saving the data
    index = str(n)+"_"+str(d)+"_"+str(eigen_decay)+"_"+str(cova_noise)+"_"+str(noise)
    np.save("Dataset/Synthetic/train_x_"+index+"_.npy",train_x)
    np.save("Dataset/Synthetic/train_y_"+index+"_.npy",train_y)
    #Making directory to store results for this dataset
    directory = "Results/Synthetic/"+index    
    if not os.path.exists(directory):
        os.makedirs(directory)
        directory_Error = directory+"/Error"
        directory_Figure = directory+"/Figure"
        directory_History = directory+"/History"
        os.makedirs(directory_Error)
        os.makedirs(directory_Figure)
        os.makedirs(directory_History)
    return train_x, train_y, optimal


# synthetic LSR data with n features, d dimensions, 
# eigenvalue decay as decay, magnitude of the covariance 
# of the data as cova_noise and additive noise as noise 
def synthetic_data_loader(n,d,decay,cova_noise,noise):
    path = "Dataset/Synthetic/train_x_"+str(n)+"_"+str(d)+"_"+str(decay)+"_"+str(cova_noise)+"_"+str(noise)+"_.npy"
    if os.path.isfile(path):
        x = np.load("Dataset/Synthetic/train_x_"+str(n)+"_"+str(d)+"_"+str(decay)+"_"+str(cova_noise)+"_"+str(noise)+"_.npy")
        y = np.load("Dataset/Synthetic/train_y_"+str(n)+"_"+str(d)+"_"+str(decay)+"_"+str(cova_noise)+"_"+str(noise)+"_.npy")
        optimal = np.array(np.linalg.lstsq(x,y,rcond=None)[0])
    else:
        print("Generating Synthetic Data....")
        x,y,optimal = data_maker(n,d,decay,cova_noise,noise)
    index = str(n)+"_"+str(d)+"_"+str(decay)+"_"+str(cova_noise)+"_"+str(noise)
    return x, y, optimal, index

def index_parser(index):
    index = index.split("_")
    C = int(index[0])
    P = int(index[1])
    N = int(index[2])
    step = float(index[3])
    b = int(index[4])
    return C, P, N, step, b


    
    

