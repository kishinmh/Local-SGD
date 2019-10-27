###############################################################
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import sys
import random
import matplotlib.pyplot as plt
from algorithm import *
from helper import *
np.random.seed(0)
##########################Loading Dataset############################

#data = "CPU" #Least Squares Regression Dataset
#data = "Slice" #Least Squares Regression Dataset
#data = "E2006" #Least Squares Regression Dataset
#data = "Year" #Least Squares Regression Dataset
#data = "Higgs" #Logistic Squares Regression Dataset
#data = "Splice" #Logistic Regression Dataset
#data = "Gisette" #Logistic Regression Dataset
#data = "Default" #Logistic Regression Dataset, It is an unbalanced data-set
#data = "Covtype" #Logistic Regression Dataset
#data = "ijcnn1" #Logistic Regression Dataset, It is an unbalanced data-set
data = "Madelon" #Logistic Regression Dataset

using_synthetic_data = 0
if using_synthetic_data:
    train_x, train_y, optimal, index = synthetic_data_loader(10000,1000,3,100,1)
else:
    train_x, train_y, optimal = data_loader(data)
d = len(train_x[0]) # Dimension of the features
print "Length of the Training Data:" + str(len(train_y))
print "Feature Dimension: " + str(d)

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

#steps = [0.4*step,0.5*step,0.6*step,0.7*step,0.8*step,0.9*step,step,1.1*step,1.2*step] 
steps = [0.9*step,step,1.1*step,1.2*step]
Error = {}

for step in steps:
    Error[step] = 0
    print "For stepsize: "+str(step)
    

    ##################The Main Process##################################
    for rep in range(nrep):
        print "In repetetion no:" + str(rep+1)+"/"+str(nrep)

        ####################Allocating Memory#############################

        random_state = np.arange(P) # fixed random seeds for all the machines
        history = np.zeros((C,N,d)) # Store the average over all the machines

        if optima:
            theta = [optimal]*P # When we initialise at the optimal (found through inverse)
        else:
            theta = [np.random.normal(0,1,d)]*P # When we initialise at a random point

        error = np.zeros((nrep,C*N)) # Store the error for the current estimator 
        avg_error = np.zeros((nrep,C*N)) # Store the error for the average estimator or tail averaged estimator
        
        for t in range(C): # communication phases 
            print("In epoch number: " + str(t))
            #It is the step where we parallelize the task into "P" machines
            #using "n_jobs" cores. The R.H.S. returns the history of parameters overl all
            # the machines for phase t of communication. Note the random state rng passed
            # into the function, which ensures the experiments are deterministic. A bug was
            # found in the package which conerns me about how good is it. Note though, that the 
            # backend, multiprocessing is a standard python library.   
            temp = Parallel(n_jobs=jobs, backend="multiprocessing")(delayed(LSR)(train_x, train_y, theta[i], N, b, step, random_state[i],0) for i in range(P))
            #passing the average of all the last iterates into the next phase
            theta = np.mean(np.array(temp)[:,-1],0)
            theta = np.tile(theta,(P,1))
            history[t]=np.mean(temp,axis=0)#Store just the average of all workers
        history = history.reshape(C*N,d) #reshaping the matrix into a straight array.
        final = np.mean(history,axis=0)
        #Reporting the error averaged over all the repetetions
        pred = np.dot(train_x,final)
        Error[step] = (rep*Error[step]+nl(train_x,train_y,optimal,pred))/(rep+1)
    
##########################Plotting the curves directly#######################
y=[]
for key in steps:
    y.append(Error[key])

plt.figure(3)
plt.style.use("ggplot")
plt.plot(steps,np.log2(y))
plt.xlabel("Learning Rate")
plt.ylabel("Final Error")
if using_synthetic_data:
    plt.suptitle("LSR: Synthetic "+ index +" Dataset, Final Estimator")
    plt.legend()
    plt.savefig("Results/Synthetic/"+index+"/Figure/tune_"+local_SGD_index+"_.pdf",dpi=1000)
else:    
    plt.suptitle("LSR: "+data+" Dataset, Final Estimator")
    plt.legend()
    plt.savefig("Results/"+data+"/Figure/tune_"+local_SGD_index+"_.pdf",dpi=1000)
plt.close()
