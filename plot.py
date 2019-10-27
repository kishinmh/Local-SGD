########################Loading Dependencies##############################
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from helper import *
np.random.seed(0)
##########################Loading Dataset for error####################

data = "CPU" #Least Squares Regression Dataset
#data = "Slice" #Least Squares Regression Dataset
#data = "E2006" #Least Squares Regression Dataset
#data = "Year" #Least Squares Regression Dataset
#data = "Higgs" #Logistic Squares Regression Dataset
#data = "Splice" #Logistic Regression Dataset
#data = "Gisette" #Logistic Regression Dataset
#data = "Default" #Logistic Regression Dataset, It is an unbalanced data-set
#data = "Covtype" #Logistic Regression Dataset
#data = "ijcnn1" #Logistic Regression Dataset, It is an unbalanced data-set
#data = "Madelon" #Logistic Regression Dataset

using_synthetic_data = 0
if using_synthetic_data:
    train_x, train_y, optimal, index = synthetic_data_loader(10000,1000,3,100,1)
else:
    train_x, train_y, optimal = data_loader(data)
optima = 1
Error = {}
Error_std = {}
Avg_Error = {}
Avg_Error_std = {}
Time = {}

#To identify the local SGD run
indices = [("1_1_1000_0.07_4096",1),("1_32_1000_0.07_128",1),("10_32_100_0.07_128",1),("100_32_10_0.07_128",1)]
for alg in indices:
    C,P,N,step,b = index_parser(alg[0])
    #The mini batch case
    if P ==1 and C==1:
        C=N
        N=1
        P=b
        b=1
    
    local_SGD_index = alg[0]
    freq = alg[1]

    if using_synthetic_data:
        if optima:
            name_error = "Results/Synthetic/"+index+"/Error/Opt_Error_"+local_SGD_index+"_.npy"
            name_avg_error = "Results/Synthetic/"+index+"/Error/Opt_Avg_Error_"+local_SGD_index+"_.npy"
        else:
            name_error = "Results/Synthetic/"+index+"/Error/Error_"+local_SGD_index+"_.npy"
            name_avg_error = "Results/Synthetic/"+index+"/Error/Avg_Error_"+local_SGD_index+"_.npy"
    else:
        if optima:
            name_error = "Results/"+data+"/Error/Opt_Error_"+local_SGD_index+"_.npy"
            name_avg_error = "Results/"+data+"/Error/Opt_Avg_Error_"+local_SGD_index+"_.npy"
        else:
            name_error = "Results/"+data+"/Error/Error_"+local_SGD_index+"_.npy"
            name_avg_error = "Results/"+data+"/Error/Avg_Error_"+local_SGD_index+"_.npy"

    error = np.mean(np.load(name_error), axis=0)
    error_std = np.std(np.load(name_error), axis=0)
    avg_error = np.mean(np.load(name_avg_error),axis=0)
    avg_error_std = np.std(np.load(name_avg_error),axis=0)
    Time[N] = P*b*freq*(np.arange(len(error))+1)
    Error[N] = error
    Error_std[N] = error_std
    Avg_Error[N] = avg_error
    Avg_Error_std[N] = avg_error_std

########################Making Error Plots###################################
if optima == 0:
    plt.figure(0)
    plt.style.use("ggplot")
    for key in np.sort(list(Error.keys())):
        plt.plot(np.log2(Time[key][0:800]),np.log2(Error[key][0:800]),label="N="+str(key))
        #plt.fill_between(Time[key],Error[key]-2*Error_std[key], Error[key]+2*Error_std[key], alpha=0.2)
    plt.xlabel("Gradients Accessed")
    plt.ylabel("Normalized Objective Error")
    if using_synthetic_data:
        plt.suptitle("LSR: Synthetic "+ index +" Dataset, Current Estimator")
        plt.legend()
        plt.savefig("Results/Synthetic/"+index+"/Figure/Loss_"+str(P)+"_.pdf",dpi=1000)
    else:    
        plt.suptitle("LSR: "+ data +" Dataset, Current Estimator")
        plt.legend()
        plt.savefig("Results/"+data+"/Figure/Loss_"+str(P)+"_"+str(b)+"_.pdf",dpi=1000)
    plt.close()

    plt.figure(1)
    plt.style.use("ggplot")
    for key in np.sort(list(Avg_Error.keys())):
        plt.plot(np.log2(Time[key][0:800]),np.log2(Avg_Error[key][0:800]),label="N="+str(key))
        #plt.fill_between(Time[key],Avg_Error[key]-2*Avg_Error_std[key],Avg_Error[key]+2*Avg_Error_std[key],alpha=0.2)
    plt.xlabel("Gradients Accesed")
    plt.ylabel("Normalized Objective Error")
    if using_synthetic_data:
        plt.suptitle("LSR: Synthetic "+ index +" Dataset, Average Estimator")
        plt.legend()
        plt.savefig("Results/Synthetic/"+index+"/Figure/Loss_"+str(P)+"_.pdf",dpi=1000)
    else:    
        plt.suptitle("LSR: "+ data +" Dataset, Average Estimator")
        plt.legend()
        plt.savefig("Results/"+data+"/Figure/Avg_Loss_"+str(P)+"_"+str(b)+"_.pdf",dpi=1000)
    plt.close()
else:
    plt.figure(0)
    plt.style.use("ggplot")
    for key in np.sort(list(Error.keys())):
        plt.plot(np.log2(Time[key][0:800]),np.log2(Error[key][0:800]),label="N="+str(key))
        #plt.fill_between(Time[key],Error[key]-2*Error_std[key], Error[key]+2*Error_std[key], alpha=0.2)
    plt.xlabel("Gradients Accessed")
    plt.ylabel("Normalized Objective Error")
    if using_synthetic_data:
        plt.suptitle("LSR: Synthetic "+ index +" Dataset, Current Estimator")
        plt.legend()
        plt.savefig("Results/Synthetic/"+index+"/Figure/Loss_"+str(P)+"_.pdf",dpi=1000)
    else:    
        plt.suptitle("LSR: "+ data +" Dataset, Current Estimator")
        plt.legend()
        plt.savefig("Results/"+data+"/Figure/Opt_Loss_"+str(P)+"_"+str(b)+"_.pdf",dpi=1000)
    plt.close()

    plt.figure(1)
    plt.style.use("ggplot")
    for key in np.sort(list(Avg_Error.keys())):
        print(key)
        print(np.log2(Avg_Error[key][-10:]))
        plt.plot(np.log2(Time[key]),np.log2(Avg_Error[key]),label="N="+str(key))
        #plt.fill_between(Time[key],Avg_Error[key]-2*Avg_Error_std[key],Avg_Error[key]+2*Avg_Error_std[key],alpha=0.2)
    plt.xlabel("Gradients Accesed")
    plt.ylabel("Normalized Objective Error")
    if using_synthetic_data:
        plt.suptitle("LSR: Synthetic "+ index +" Dataset, Average Estimator")
        plt.legend()
        plt.savefig("Results/Synthetic/"+index+"/Figure/Loss_"+str(P)+"_.pdf",dpi=1000)
    else:    
        plt.suptitle("LSR: "+ data +" Dataset, Average Estimator")
        plt.legend()
        plt.savefig("Results/"+data+"/Figure/Opt_Avg_Loss_"+str(P)+"_"+str(b)+"_.pdf",dpi=1000)
    plt.close()

##########################Loading Dataset for squared error####################

# indices = [("10_64_100_100.0_1",1)]


# SD = {}
# SD_17 = {}
# Time = {}

# #To identify the local SGD run
# for alg in indices:
#     C,P,N,step,b = index_parser(alg[0])
#     local_SGD_index = alg[0]
#     freq = alg[1]
#     if P >1:
#         if using_synthetic_data:
#             if optima:
#                 name_sd_17 = "Results/Synthetic/"+index+"/History/17_Opt_History_"+local_SGD_index+"_.npy"
#                 name_sd = "Results/Synthetic/"+index+"/History/Opt_History_"+local_SGD_index+"_.npy"
#             else:
#                 name_sd_17 = "Results/Synthetic/"+index+"/History/17_History_"+local_SGD_index+"_.npy"
#                 name_sd = "Results/Synthetic/"+index+"/History/History_"+local_SGD_index+"_.npy"
#         else:
#             if optima:
#                 name_sd_17 = "Results/"+data+"/History/17_Opt_History_"+local_SGD_index+"_.npy"
#                 name_sd = "Results/"+data+"/History/Opt_History_"+local_SGD_index+"_.npy"
#             else:
#                 name_sd_17 = "Results/"+data+"/History/17_History_"+local_SGD_index+"_.npy"
#                 name_sd = "Results/"+data+"/History/History_"+local_SGD_index+"_.npy"
#         sd_17 = np.mean(np.load(name_sd_17),axis=0)
#         sd = np.mean(np.load(name_sd),axis=0)
#         Time[N] = P*b*freq*(np.arange(len(sd))+1)/float(len(train_x))  
#         SD[N] = sd
#         SD_17[N] = sd_17 

# ########################Making SD Plots###################################

# plt.figure(2)
# plt.style.use("ggplot")
# for key in np.sort(list(SD.keys())):
#     plt.plot(Time[key],SD[key],label="N="+str(key))
#     plt.plot(Time[key],SD_17[key],label="Worker N="+str(key))
# plt.xlabel("Epochs")
# plt.ylabel("Squared Distance From Optimal")
# if using_synthetic_data:
#     plt.suptitle("LSR: Synthetic "+ index +" Dataset, Current Estimator")
#     plt.legend()
#     plt.savefig("Results/Synthetic/"+index+"/Figure/SD_"+str(P)+"_.pdf",dpi=1000)
# else:    
#     plt.suptitle("LSR: "+ data +" Dataset, Current Estimator")
#     plt.legend()
#     plt.savefig("Results/"+data+"/Figure/SD_"+str(P)+"_"+str(b)+"_.pdf",dpi=1000)
# plt.close()


    
    



