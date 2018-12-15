
# coding: utf-8

# # Project 3 on Classification and GD

# In[1]:


import pandas as pd
import scipy as sci
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import time


# ## Task 1
# 

# In[2]:


# task1
x_A = pd.read_csv('/Users/jianwenliu/MLPP3/PP3data/A.csv').values
x_B = pd.read_csv('/Users/jianwenliu/MLPP3/PP3data/B.csv').values
x_USPS = pd.read_csv('/Users/jianwenliu/MLPP3/PP3data/USPS.csv').values

y_A = pd.read_csv('/Users/jianwenliu/MLPP3/PP3data/labels-A.csv').values
y_B = pd.read_csv('/Users/jianwenliu/MLPP3/PP3data/labels-B.csv').values
y_USPS = pd.read_csv('/Users/jianwenliu/MLPP3/PP3data/labels-USPS.csv').values

x_irlstest = pd.read_csv('/Users/jianwenliu/MLPP3/PP3data/irlstest.csv').values
y_irlstest = pd.read_csv('/Users/jianwenliu/MLPP3/PP3data/labels-irlstest.csv').values


# In[3]:


#prepare for the train/test data
def data_prepare(x,y,ratio):
   
    N = len(y)
    index = list(range(N))
    np.random.shuffle(index) # Randomly shuffle the indices
    
    
    test_index = index[0:int(N/3.0)]
    train_index = index[int(N/3):int((1/3 + ratio * 2/3) * N)] # ratio is size rate of training examples
    
    xtrain = x[train_index]
    ytrain = y[train_index]
    xtest = x[test_index]
    ytest = y[test_index]
        
    return xtrain, ytrain, xtest, ytest


# In[4]:


#the generaltive model
def G_M(xdata,ydata,ratio):
    
    xtrain, ytrain, xtest, ytest=data_prepare(xdata,ydata,ratio)
    
    #train process
    #find the location of 1 and 0
    c1_loc = np.where(ytrain == 1)[0]
    c0_loc = np.where(ytrain == 0)[0]
    
    #split data by its label
    xtrain_1 = xtrain[c1_loc]
    xtrain_0 = xtrain[c0_loc]
    
    #mean of training data on 1 and 0
    mu_1 = xtrain_1.mean(0)
    mu_0 = xtrain_0.mean(0)
    
    #N for sample
    N_1 = xtrain_1.shape[0]
    N_0 = xtrain_0.shape[0]
    N = N_1 + N_0
    p_1 = N_1/N
    p_0 = N_0/N
    
    #M for features
    M = xtrain.shape[1]
    # calculate the sigma and its inverse for w
    Sigma1 = np.zeros([M,M])
    Sigma0 = np.zeros([M,M])
    for n in range(N_1):
        Sigma1 += np.outer(xtrain_1[n] - mu_1, np.transpose(xtrain_1[n] - mu_1))
    Sigma1 = Sigma1 / N_1
    for n in range(N_0):
        Sigma0 += np.outer(xtrain_0[n] - mu_0, np.transpose(xtrain_0[n] - mu_0))
    Sigma0 = Sigma0 / N_0
    Sigma = p_1*Sigma1 + p_0 * Sigma0
    
    Sigma_inv = np.linalg.inv(Sigma)
    
    #calculate w and b(intercept)
    w = np.dot(Sigma_inv, mu_1 - mu_0)
    b = - 1/2 * np.dot(mu_1, np.dot(Sigma_inv, mu_1))+ 1/2 * np.dot(mu_0, np.dot(Sigma_inv, mu_0)) + np.log(p_1/p_0)
    
    #test  process
    perf = []
    for k in range(len(xtest)):
        a = np.dot(w,xtest[k]) + b
        sigmoid = 1/(1 + np.exp(-a))
        #if sigmoid is bigger than 0.5, than we justify it to 1, otherwise it is zero
        if sigmoid >= 0.5:
            perf.append(int(ytest[k] == 1))
        else:
            perf.append(int(ytest[k] == 0))
    #calculate the accuracy
    acr = (sum(perf))/len(perf)
    
    return acr


# In[5]:


#Bayesian logistic regression model
def B_Lo(xdata,ydata,ratio):
    
    #data input
    xtrain, ytrain, xtest, ytest=data_prepare(xdata,ydata,ratio)
    
    #concatenate a column of "one" to the train data
    N = xtrain.shape[0]
    onescol = np.array([[1]]*N)
   
    xtrain = np.concatenate((onescol, xtrain), axis = 1)
    M = xtrain.shape[1]

    #initial alpha
    alpha = 0.1
    w = np.array([[0]]*M)
    criterion = 1
    n = 1
    
    
    while criterion>10**-3 and n<100:
    
        w_before = w

        a = xtrain.dot(w_before)
        sig = 1.0/(1 + np.exp(-a))
        r = sig*(1-sig)
        R = np.diag(r.ravel())
        I=np.eye(M)
        A = alpha * I + xtrain.transpose().dot(np.dot(R,xtrain))
        B = xtrain.transpose().dot(sig - ytrain) + alpha*w_before
        w_upd = w_before - np.dot(np.linalg.inv(A),B)
        
        criterion = np.linalg.norm(w_upd - w_before) / np.linalg.norm(w_before)
        w=w_upd
        n =n+1
    
    a = xtrain.dot(w)
    y = 1.0/(1 + np.exp(-a))
    Sigma_inv = I*alpha
    for n in range(N):
        Sigma_inv += y[n]*(1-y[n]) * np.outer(xtrain[n],xtrain[n])
    Sigma = np.linalg.inv(Sigma_inv)
    
    #test the model
    #concatenate a coloum of "one" to the test data
    ones = np.array([[1]]*xtest.shape[0])
    xtest = np.concatenate((ones, xtest), axis = 1)
    perf = []
    for n in range(xtest.shape[0]):
        mu_a = xtest[n].dot(w)
        sigma_square = xtest[n].transpose().dot(Sigma.dot(xtest[n]))
        kap = (1 + np.pi * sigma_square / 8) ** (-1.0/2)
        p = 1.0/(1 + np.exp( - kap * mu_a))
        #if sigmoid is bigger than 0.5, than we justify it to 1, otherwise it is zero
        
        if p >= 0.5:
            perf.append(int(ytest[n] == 1))
        else:
            perf.append(int(ytest[n] == 0))
    #calculate the accuracy
    acr = (sum(perf))/len(perf)
    
    return acr


# In[6]:


#copare these two models with the same data size
def compare_two(xdata,ydata):
    size_domain = np.arange(0.1,1.1,0.1)
    acr_gm_mean = []
    acr_gm_std = []
    acr_bl_mean = []
    acr_bl_std = []
    #change the data size
    for size in size_domain:
        acr_gm = []
        acr_bl = []
        #repeate 30 times to generate curves
        for k in range(30):
            acr_gm.append(G_M(xdata,ydata,size))
            acr_bl.append(B_Lo(xdata,ydata,size))
        acr_gm_mean.append(np.mean(acr_gm))
        acr_gm_std.append(np.std(acr_gm))
        acr_bl_mean.append(np.mean(acr_bl))
        acr_bl_std.append(np.std(acr_bl))
    
    return size_domain, acr_gm_mean, acr_gm_std, acr_bl_mean, acr_bl_std


# In[7]:


#on data set of A
Asize_domain, Aacr_gm_mean, Aacr_gm_std, Aacr_bl_mean, Aacr_bl_std = compare_two(x_A,y_A)


# In[8]:


# plot
fig = plt.gcf()
fig.set_size_inches(20, 15)
plt.errorbar(Asize_domain, Aacr_gm_mean, yerr = Aacr_gm_std, fmt = 'g')
plt.errorbar(Asize_domain, Aacr_bl_mean, Aacr_bl_std, fmt = 'r')
plt.xlabel('train datasize')
plt.ylabel('Accuracy')
plt.title('Accuracy on A')
plt.legend(['Generative Model','BLR'])


# In[9]:


Bsize_domain, Bacr_gm_mean, Bacr_gm_std, Bacr_bl_mean, Bacr_bl_std = compare_two(x_B,y_B)


# In[10]:


fig = plt.gcf()
fig.set_size_inches(20, 15)

plt.errorbar(Bsize_domain, Bacr_gm_mean, yerr = Bacr_gm_std, fmt = 'g')
plt.errorbar(Bsize_domain, Bacr_bl_mean, Bacr_bl_std, fmt = 'r')


plt.xlabel('training data size')
plt.ylabel('Accuracy')
plt.title('Accuracy on B')
plt.legend(['Generative Model','BLR'])


# In[11]:


#Use USPS data
USPSsize_domain, USPSacr_gm_mean, USPSacr_gm_std, USPSacr_bl_mean, USPSacr_bl_std = compare_two(x_USPS,y_USPS)


# In[12]:


#plot
fig = plt.gcf()
fig.set_size_inches(20, 15)
plt.errorbar(USPSsize_domain, USPSacr_gm_mean, yerr = USPSacr_gm_std, fmt = 'g')
plt.errorbar(USPSsize_domain, USPSacr_bl_mean, USPSacr_bl_std, fmt = 'r')
plt.xlabel('training data size')
plt.ylabel('Accuracy')
plt.title('Accuracy on USPS')
plt.legend(['Generative Model','BLR'])


# ## Q & A for Task 1
# 
# Q: In your submission plot these results and discuss them:how do the algorithms perform on these datasets? Are there systematic differences? and how do the differences depend on training set size?
# 
# A: For dataset A, Firstly, we can find out that the generative model performs better than that of the Bayesian Logistic Regression. 
# As the dataset increases, both performances will increase and the BLR will finally outperforms the GM. 
# This is because because the shared covariance matrix condition for the Gaussian assumption of the generative model could hardly be satisfied for dataset A when datasize is big. As shown in the data delarement, data A is actually draw from a uniform dataset. When data size is small. this doe not hurt the assumption too much. However, as the data size increases, this unfitness of normal distribution will be showm. 
# 
# For dataset B, on contary to the case of A, the Bayesian Logisitic Regression always performs better than the generative model. This is because it is generated from multiple Gaussians with differing covariance structure, which does not fit the generative model assumption. However, the Bayesian Logistic Regression can be apllied here without constraint.
# As the dataset size of training data for B increases, both performances will increase. Particularly, the Generative model has increase more as the train size increases compared to BLR. 
# 
# For dataset USPS (bitmap),the Bayesian Logisitic Regression performs better than the generative model. This is maybe because the real data of USPS do not fit the assumption of shared covirance matrix.
# As the dataset size of training data for USPS increases, both performances will increase. Particularly, the Generative model has an obviously increase compared to BLR. 
# 
# 

# ## Task 2
# 

# In[13]:


# the newton method algorithm
def B_log_newt(x,y):
    
    #use the data prepare function to input data
    xtrain, ytrain, xtest, ytest=data_prepare(x,y,1)
    
    N = xtrain.shape[0]
    #create a coloum filled with one, then concatenate it to the train data
    onescol = np.array([[1]]*N)
    xtrain = np.concatenate((onescol, xtrain), axis = 1)
    M = xtrain.shape[1]

    
    #initial alpha
    alpha = 0.1
    w = np.array([[0]]*M)
    criterion = 1
    n = 1
    
    #initial time and a array of w
    time_lapse = [0]
    w_collect = [w]
    start = time.time()
    
    while criterion>10**-3 and n<100:
    
        w_before = w

        a = xtrain.dot(w_before)
        sig = 1.0/(1 + np.exp(-a))
        r = sig*(1-sig)
        #compute the hessian matrix for the newton method
        R = np.diag(r.ravel())
        I=np.eye(M)
        A = alpha * I + xtrain.transpose().dot(np.dot(R,xtrain))
        B = xtrain.transpose().dot(sig - ytrain) + alpha*w_before
        w_upd = w_before - np.dot(np.linalg.inv(A),B)#newton
        
        criterion = np.linalg.norm(w_upd - w_before) / np.linalg.norm(w_before)
        w=w_upd
        n =n+1
        #record the time
        time_lapse.append(time.time()-start)
        w_collect.append(w)
    
    #test the model 
    #prepare a accuracy to collect 
    acr_collect = []
    #concatenate a column of "one" to the test data
    ones = np.array([[1]]*xtest.shape[0])
    xtest = np.concatenate((ones, xtest), axis = 1)
    
    for w in w_collect:
        a = xtrain.dot(w)
        sigmo = 1/(1 + np.exp(-a))
        Sigma_inv = np.eye(M)*alpha
        for n in range(N):
            Sigma_inv += sigmo[n]*(1-sigmo[n]) * np.outer(xtrain[n],xtrain[n])
        Sigma = np.linalg.inv(Sigma_inv)
        
        #initiate a performance for collect
        perf = []
        for n in range(xtest.shape[0]):
            mu_a = xtest[n].dot(w)
            sigma_a_squared = xtest[n].transpose().dot(Sigma.dot(xtest[n]))
            kappa = (1 + np.pi * sigma_a_squared / 8) ** (-1.0/2)
            p = 1.0/(1 + np.exp( - kappa * mu_a))
            #if sigmoid is bigger than 0.5, than we justify it to 1, otherwise it is zero
            if p >= 0.5:
                perf.append(int(ytest[n] == 1))
            else:
                perf.append(int(ytest[n] == 0))
        acr = (sum(perf))/len(perf)

        acr_collect.append(acr)
    
    return w_collect, time_lapse, acr_collect


# In[14]:


w_newt1, time_newt1, acr_newt1 = B_log_newt(x_A,y_A)


# In[15]:


def B_log_gradAsc(x,y,learnrate):
    
    #data prepare for the model
    N = len(y)
    index = list(range(N))
    test_index = index[0:int(N/3)]
    train_index = index[int(N/3):int(N)] 
    xtrain = x[train_index]
    ytrain = y[train_index]
    xtest = x[test_index]
    ytest = y[test_index]
    
    #concatenate a column filled with "one" to the feature(test)
    N = xtrain.shape[0]
    onescol = np.array([[1]]*N)
    xtrain = np.concatenate((onescol, xtrain), axis = 1)
    M = xtrain.shape[1]

    #initial alpha
    alpha = 0.1
    w = np.array([[0]]*M)
    criterion = 1
    n = 1
    
    #initiate the time and a space for collecting w
    time_lapse = [0]
    w_collect = [w]
    start = time.time()
    
    #set up the stopping criterion as shown in task
    while criterion>10**-3 and n<6000:
    
        w_before = w

        #the update of w with gradient ascent
        a = xtrain.dot(w_before)
        sigmo = 1/(1 + np.exp(-a))
        #use the learnrate in the gradient ascent
        w_upd = w_before - learnrate * (xtrain.transpose().dot(sigmo - ytrain) + alpha * w_before)#gradian ascent
        
        #compute the stopping criterion
        criterion = np.linalg.norm(w_upd - w_before) / np.linalg.norm(w_before)
        w=w_upd
        n += 1
        if n%50 == 0: # run time/50 iterations
            time_lapse.append(time.time()-start)
            w_collect.append(w)
    
     #test the model
    #concatenate a column filled with "one" to the feature(test)
    ones = np.array([[1]]*xtest.shape[0])
    xtest = np.concatenate((ones, xtest), axis = 1)
    I = np.eye(M)
    #initiate the array to collect the accuracy
    acr_collect = []
    for w in w_collect:
        a = xtrain.dot(w)
        sigmo = 1/(1 + np.exp(-a))
        Sigma_inv = alpha*I
        for n in range(N):
            Sigma_inv += sigmo[n]*(1-sigmo[n]) * np.outer(xtrain[n],xtrain[n])
        Sigma = np.linalg.inv(Sigma_inv) 
    
        #initiate the array to collect the perfomance to get the accuracy
        perf = []
        for n in range(xtest.shape[0]):
            mu_a = xtest[n].dot(w)
            sigma_square = xtest[n].transpose().dot(Sigma.dot(xtest[n]))
            kap = (1 + np.pi * sigma_square / 8) ** (-1/2)
            p = 1/(1 + np.exp( - kap * mu_a))
            #the 
            if p >= 0.5:
                perf.append(int(ytest[n] == 1))
            else:
                perf.append(int(ytest[n] == 0))
        acr = (sum(perf))/len(perf)

        acr_collect.append(acr)
    
    return w_collect, time_lapse, acr_collect


# In[16]:


#use a function to compute the avrage run time (for three iterations) on algorithms respectively
def avg_time(x,y,learnrate):
    #run the model three times for newton and gradien ascent
    w_newt1, time_newt1, acr_newt1 = B_log_newt(x,y)
    w_newt2, time_newt2, acr_newt2 = B_log_newt(x,y)
    w_newt3, time_newt3, acr_newt3 = B_log_newt(x,y)
    
    w_ga1, time_ga1, acr_ga1= B_log_gradAsc(x,y,learnrate)
    w_ga2, time_ga2, acr_ga2= B_log_gradAsc(x,y,learnrate)
    w_ga3, time_ga3, acr_ga3= B_log_gradAsc(x,y,learnrate)
    
    #average on three
    time_newt=(np.array(time_newt1)+np.array(time_newt2)+np.array(time_newt3))/3
    acr_newt=(np.array(acr_newt1)+np.array(acr_newt2)+np.array(acr_newt3))/3
    
    time_ga=(np.array(time_ga1)+np.array(time_ga2)+np.array(time_ga3))/3
    acr_ga=(np.array(acr_ga1)+np.array(acr_ga2)+np.array(acr_ga3))/3
    
    return time_newt,acr_newt,time_ga,acr_ga


# In[17]:


fig = plt.gcf()
fig.set_size_inches(20, 15)
time_newtA,acr_newtA,time_gaA,acr_gaA=avg_time(x_A,y_A,0.001)
plt.plot(time_newtA,acr_newtA,'g-*')
plt.plot(time_gaA,acr_gaA,'r-^')
plt.xlabel("Running Time")
plt.ylabel("Accuracy")
plt.title("Accuracy of Newton Method and Gradiant Ascent on A")
plt.legend(["Newton Method","Gradient Ascent"])


# In[18]:


fig = plt.gcf()
fig.set_size_inches(20, 15)
time_newtUSPS,acr_newtUSPS,time_gaUSPS,acr_gaUSPS=avg_time(x_USPS,y_USPS,0.001)
plt.plot(time_newtUSPS,acr_newtUSPS,'g-*')
plt.plot(time_gaUSPS,acr_gaUSPS,'r-^')
plt.xlabel("Running Time")
plt.ylabel("Accuracy")
plt.title("Accuracy of Newton Method and Gradiant Ascent on USPS")
plt.legend(["Newton Method","Gradient Ascent"])


# ## Q&A for Task 2
# 
# Q:In your submission plot these results and discuss them: how do the algorithms perform on these datasets? are there systematic differences? and how do the differences depend on data set characteristics? (Compare the Diffrence of Algorithm and Discussion for the reason?)
# 
# A: For data A, the Newton method took longer time for itration than that of Gradient Ascent. However, Newton method need much less iterations to converge than Gradient Ascent. 
# In total, Newton method still has much less time than Gradient Ascent. This is because the learning rate of the Gradient Ascent will keep the same with the number of iterations. This leads the algorithm to be very slow especially at the earliest steps. Meanwhile, Newton method is much faster. Even though we need time to compute the exact form of the hessian matrix. This Matrix will help the algorith to converge much faster for it decides what the learning rate is based on its current location. 
# 
# For data B, similar to the A, the Newton method took longer time for itration than that of Gradient Ascent. This is obvious easpecially in the first iteration. However, Newton method need much less iterations to converge than Gradient Ascent. In total, Newton method took longer time than Gradient Ascent. In case of USPS, we can find out that both algorithms converges very quickily. This shows that both newton method and Gradient Ascent have a good effect on this data.
# 
# To sum up, Newton method may cost more time to compute the Hessian matrix, but if it can be overcome camparatively easily, it will converges faster than the gradient ascent.

# ## Extra Credit
# 

# In[19]:


def B_log_linesearch(x,y,initiallearnrate):
    
    #data preparation
    N = len(y)
    index = list(range(N))
    xtrain = x[index]
    ytrain = y[index]
    xtest = x[index]
    ytest = y[index]
    
    #concatenate a column filled with "one" to the feature(t)
    N = xtrain.shape[0]
    onescol = np.array([[1]]*N)
    xtrain = np.concatenate((onescol, xtrain), axis = 1)
    M = xtrain.shape[1]

    #initial alpha
    alpha = 0.1
    w = np.array([[0]]*M)
    criterion = 1
    n = 1
    
   #collect w
    w_collect = [w]

    
    # set up the stopping criterion
    while criterion>10**-3 and n<6000:
    
        w_before = w

        a = xtrain.dot(w_before)
        sigmo = 1/(1 + np.exp(-a))
        
        learnrate=initiallearnrate
        beta = 0.5
        #T for the true or false condition
        T = 1
        
        #the line search update
        while T:
            f = -np.log(sigmo**ytrain*(1-sigmo)**(1-ytrain)).sum()
            w_prime = w_before - learnrate * (xtrain.transpose().dot(sigmo - ytrain))
            a_prime = xtrain.dot(w_prime)
            y_prime = 1/(1 + np.exp(-a_prime))
            f_prime = -np.log(y_prime**ytrain*(1-y_prime)**(1-ytrain)).sum()
            if f_prime - f > -learnrate/2*(xtrain.transpose().dot(sigmo - ytrain)).transpose().dot(xtrain.transpose().dot(sigmo - ytrain)):
                learnrate = beta * learnrate
            else:
                T = 0
        
        #print the learn rate for every iteration
        print ("the learn rate is :", learnrate)
        
        #gradiant
        w_upd = w_before - learnrate * (xtrain.transpose().dot(sigmo - ytrain) + alpha * w_before)
        criterion = np.linalg.norm(w_upd - w_before) / np.linalg.norm(w_before)
        w=w_upd
        n += 1
        w_collect.append(w)

        # return iteration and w to see how they varies
    return n, w_collect


# In[20]:


n,w=B_log_linesearch(x_irlstest,y_irlstest,1)


# In[21]:


#the iteration number
print ("the iteration is : ", n)


# In[22]:


#the array variance for w
print ("the array for w is : ", w)


# ## Q&A for Extra Credit
# 
# Q: Implement line search and evaluate its effect on the success and convergence speed of gradient ascent.
# 
# A: With line search algorithm, we can see that the learning rate is varying with every iteration. The line search algorithm needs much less iterations to converge than basic gradient ascent with fixed learning rate. In total, the line search algorithm also converges much faster than the basic gradient ascent. This is because it addapt its learrate in every iteration.
