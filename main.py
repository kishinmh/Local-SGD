###############################################################
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import sys
import random
from algorithm import *
from helper import *
np.random.seed(0)
#################################################################

#data = "CPU" #Least Squares Regression Dataset
#data = "Slice" #Least Squares Regression Dataset
#data = "E2006" #Least Squares Regression Dataset
data = "Year" #Least Squares Regression Dataset
#data = "Higgs" #Logistic Squares Regression Dataset
#data = "Splice" #Logistic Regression Dataset
#data = "Gisette" #Logistic Regression Dataset
#data = "Default" #Logistic Regression Dataset, It is an unbalanced data-set
#data = "Covtype" #Logistic Regression Dataset
#data = "ijcnn1" #Logistic Regression Dataset, It is an unbalanced data-set
#data = "Madelon" #Logistic Regression Dataset

if data in ["Splice", "Gisette", "Default", "Higgs", "Covtype", "ijcnn1", "Madelon"]:
    regress=0
else:
    regress=1

using_synthetic_data = 0
if using_synthetic_data:
    train_x, train_y, optimal, index = synthetic_data_loader(10000,1000,3,100,1)
else:
    train_x, train_y, optimal = data_loader(data)
d = len(train_x[0]) # Dimension of the features
print("Length of the Training Data:" + str(len(train_y)))
print("Feature Dimension: " + str(d))

##################Local SGD paramters#############################
C = int(sys.argv[1]) # Number of communication rounds
P = int(sys.argv[2]) # Number of machines
N = int(sys.argv[3]) # Array storing internal steps for different rounds
step = float(sys.argv[4]) # step length
b= int(sys.argv[5]) # Internal batch size
jobs = int(sys.argv[6]) # number of cores to parallelize over
optima = 0 # if to begin at the optimal solution or a random point
nrep = 1 # Number of repetetions of the algorithm
local_SGD_index = str(C)+"_"+str(P)+"_"+str(N)+"_"+str(step)+"_"+str(b)
freq = 1 #frequency of sampling the error
final_loss = 0
decay=1
error = np.zeros((nrep,int(C*N/freq))) # Store the error for the current estimator 
avg_error = np.zeros((nrep,int(C*N/freq))) # Store the error for the average estimator or tail averaged estimator
##################The Main Process##################################
for rep in range(nrep):
    print("In repetetion no:" + str(rep+1)+"/"+str(nrep))
    
    ####################Allocating Memory#############################
    random_state = np.arange(P) # fixed random seeds for all the machines
    
    if P>16:
        history_17 = np.zeros((C,N,d)) # history of a specific worker
    history = np.zeros((C,N,d)) # Store the average over all the machines

    if optima:
        theta = [optimal]*P # When we initialise at the optimal (found through inverse)
    else:
        theta = [np.random.normal(0,1,d)]*P # When we initialise at a random point

    if P>16:
        squared_distance_17 = np.zeros((nrep,C*N)) # Distance from the optimal
        squared_distance = np.zeros((nrep,C*N))  

    for t in range(C): # communication phases 
        print("In epoch number: " + str(t))
        #It is the step where we parallelize the task into "P" machines
        #using "n_jobs" cores. The R.H.S. returns the history of parameters overl all
        # the machines for phase t of communication. Note the random state rng passed
        # into the function, which ensures the experiments are deterministic. A bug was
        # found in the package which conerns me about how good is it. Note though, that the 
        # backend, multiprocessing is a standard python library.   
        if regress:
            temp = Parallel(n_jobs=jobs, backend="multiprocessing")(delayed(LSR)(train_x, train_y, theta[i], N, b, step, random_state[i],0) for i in range(P))
        else:
            temp = Parallel(n_jobs=jobs, backend="multiprocessing")(delayed(LOG)(train_x, train_y, theta[i], N, b, step, N*(t+1), decay, 0, random_state[i]) for i in range(P))
        #passing the average of all the last iterates into the next phase
        theta = np.mean(np.array(temp)[:,-1],axis=0)
        theta = np.tile(theta,(P,1))
        history[t]=np.mean(temp,axis=0)#Store just the average of all workers
        if P>16:
            history_17[t] = temp[17]
    history = history.reshape((C*N,d)) #reshaping the matrix into a straight array.
    if P>16:
            history_17 = history_17.reshape((C*N,d))
    final = np.mean(history,axis=0)
    pred = np.dot(train_x,final)
    if regress:
        loss = nl(train_x,train_y,optimal,pred)
        print("Error: "+str(loss))
        final_loss = (rep*final_loss + loss)/float(rep + 1)
        error[rep], avg_error[rep] = error_maker(history, optimal, train_x, train_y, freq, 0)    
    else:
        loss = 1-accuracy(train_x,train_y,final)
        #np.save("optimal.npy",final)
        print("Error: "+str(loss))
        final_loss = (rep*final_loss + loss)/float(rep + 1)
        error[rep], avg_error[rep] = error_maker_log(history, train_x, train_y, freq, 0)
        
    if P>16:
        squared_distance_17[rep] = np.linalg.norm(history_17 - np.transpose(optimal),axis=1)
        squared_distance[rep] = np.linalg.norm(history - np.transpose(optimal),axis=1)
print("Average Error was: "+str(final_loss))
################Saving Everything########################################

if using_synthetic_data:
    if optima:
        name_error = "Results/Synthetic/"+index+"/Error/Opt_Error_"+local_SGD_index+"_.npy"
        name_avg_error = "Results/Synthetic/"+index+"/Error/Opt_Avg_Error_"+local_SGD_index+"_.npy"
    else:
        name_error = "Results/Synthetic/"+index+"/Error/Error_"+local_SGD_index+"_.npy"
        name_avg_error = "Results/Synthetic/"+index+"/Error/Opt_Avg_Error_"+local_SGD_index+"_.npy"
    if P>16:
        if optima:
            name_sd_17 = "Results/Synthetic/"+index+"/History/17_Opt_History_"+local_SGD_index+"_.npy"
            name_sd = "Results/Synthetic/"+index+"/History/Opt_History_"+local_SGD_index+"_.npy"
        else:
            name_sd_17 = "Results/Synthetic/"+index+"/History/17_History_"+local_SGD_index+"_.npy"
            name_sd = "Results/Synthetic/"+index+"/History/History_"+local_SGD_index+"_.npy"
else:
    if optima:
        name_error = "Results/"+data+"/Error/Opt_Error_"+local_SGD_index+"_.npy"
        name_avg_error = "Results/"+data+"/Error/Opt_Avg_Error_"+local_SGD_index+"_.npy"
    else:
        name_error = "Results/"+data+"/Error/Error_"+local_SGD_index+"_.npy"
        name_avg_error = "Results/"+data+"/Error/Avg_Error_"+local_SGD_index+"_.npy"
    if P>16:
        if optima:
            name_sd_17 = "Results/"+data+"/History/17_Opt_History_"+local_SGD_index+"_.npy"
            name_sd = "Results/"+data+"/History/Opt_History_"+local_SGD_index+"_.npy"
        else:
            name_sd_17 = "Results/"+data+"/History/17_History_"+local_SGD_index+"_.npy"
            name_sd = "Results/"+data+"/History/History_"+local_SGD_index+"_.npy"
     
np.save(name_error,error)
np.save(name_avg_error,avg_error)
if P>16:
    np.save(name_sd,squared_distance)
    np.save(name_sd_17,squared_distance_17)