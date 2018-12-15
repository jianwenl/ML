
# coding: utf-8

# # Project 2 on Bayesian Learning Regression

# In[1]:


import numpy as np
import scipy as sci
import pandas as pd
import math
from matplotlib import pyplot as plt


# In[2]:


#data raw prepare
train_100_10 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/train-100-10.csv')
train_100_100 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/train-100-100.csv')
train_1000_100 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/train-1000-100.csv')
train_f3 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/train-f3.csv')
train_f5 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/train-f5.csv')

trainR_100_10 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/trainR-100-10.csv')
trainR_100_100 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/trainR-100-100.csv')
trainR_1000_100 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/trainR-1000-100.csv')
trainR_f3 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/trainR-f3.csv')
trainR_f5 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/trainR-f5.csv')


# In[3]:


test_100_10 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/test-100-10.csv')
test_100_100 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/test-100-100.csv')
test_1000_100 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/test-1000-100.csv')
test_f3 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/test-f3.csv')
test_f5 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/test-f5.csv')

testR_100_10 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/testR-100-10.csv')
testR_100_100 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/testR-100-100.csv')
testR_1000_100 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/testR-1000-100.csv')
testR_f3 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/testR-f3.csv')
testR_f5 = pd.read_csv('/Users/jianwenliu/MLPP2/pp2data/testR-f5.csv')


# ## Task 1

# In[4]:


# the linear regression model
lamrange=np.arange(1,150)

def l_r(traininputs,traintargets,testinputs,testtargets):
    
    xtrain=traininputs.values
    ttrain=traintargets.values
    xtest=testinputs.values
    ttest=testtargets.values
    
    mse_vec_tr=[]
    mse_vec_te=[]
    
    #loop on lambda
    for lam in lamrange:
        lambd = np.diag(np.ones(xtrain.shape[1])*lam)
        w=np.dot(np.dot(np.linalg.inv(lambd+np.dot(xtrain.transpose(),xtrain)),xtrain.transpose()),ttrain)
        mse_tr=np.sum(((np.dot(xtrain,w)-ttrain)**2))/ttrain.shape[0]
        mse_vec_tr.append(mse_tr)
        
        mse_te=np.sum(((np.dot(xtest,w)-ttest)**2))/ttest.shape[0]
        mse_vec_te.append(mse_te)
    
    #opimun_lamb_vec=lamrange[np.argmin(mse_vec_te)]
    return mse_vec_tr,mse_vec_te#,opimun_lamb_vec


# In[5]:


# calculate the mse for 10_100 data
mse_vec_tr10,mse_vec_te10=l_r(train_100_10,trainR_100_10,test_100_10,testR_100_10)
mse_true = [3.78] * len(lamrange)

#plot the mse for 10_100 data
plt.plot(lamrange, mse_vec_tr10)
plt.plot(lamrange, mse_vec_te10)
plt.plot(lamrange, mse_true)

# calculate the optimun lambda for 10_100 data
opimun_lamb_10=lamrange[np.argmin(mse_vec_te10)]

print ('the optimum lambda is', opimun_lamb_10)

plt.xlabel("range of lambda")
plt.ylabel("mse")
plt.legend(["train", "test", "true"])
plt.show()


print ('the minimum mse is',min(mse_vec_te10))


# In[6]:


# calculate the mse for 100_100 data
mse_vec_tr100,mse_vec_te100=l_r(train_100_100,trainR_100_100,test_100_100,testR_100_100)
mse_true = [3.78] * len(lamrange)


plt.plot(lamrange, mse_vec_tr100)
plt.plot(lamrange, mse_vec_te100)
plt.plot(lamrange, mse_true)

# calculate the optimun lambda for 100_100 data
opimun_lamb_100=lamrange[np.argmin(mse_vec_te100)]
plt.xlabel("range of lambda")
plt.ylabel("mse")
plt.legend(["train", "test", "true"])
plt.show()

print ('the optimum lambda is', opimun_lamb_100)

print ('the minimum mse is',min(mse_vec_te100))


# In[7]:


# calculate the mse for 1000_100 data
mse_vec_tr1000,mse_vec_te1000=l_r(train_1000_100,trainR_1000_100,test_1000_100,testR_1000_100)
mse_true = [4.015] * len(lamrange)


plt.plot(lamrange, mse_vec_tr1000)
plt.plot(lamrange, mse_vec_te1000)
plt.plot(lamrange, mse_true)

# calculate the optimun lambda for 1000_100 data
opimun_lamb_1000=lamrange[np.argmin(mse_vec_te1000)]
plt.xlabel("range of lambda")
plt.ylabel("mse")
plt.legend(["train", "test", "true"])
plt.show()

print ('the optimum lambda is', opimun_lamb_1000)

print ('the minimum mse is',min(mse_vec_te1000))


# ## Q&A for Task 1
# 
# Q:In your report provide the results/plots and discuss them: Why can't the training set MSE be used to select lambda ? How does lambda affect error on the test set? How does the choice of the optimal lambda vary with the number of features and number of examples? How do you explain these variations?
#     
# A:The reason that we can not use MSE of training set to calculate lambda is that the the MSE of trainign data is alwasy increasing with the increase of lambda. This monotonic function makes it imcapable of measuring optimal lambda.
# 
# The MSE of trainign data first decrease and then increase ultimately. This makes a optimun point of the lambda, which can be used to measure optimun lambda.
# 
# when we fix the number of variables to 100, as the feature increses from 10 to 100, the optimun lambda also increase from 8 to 22.
# when we fix the number of variables to 100, as the feature increses from 100 to 1000, the optimun lambda also increase from 22 to 27.
# When we introduce the lambda, it becomes a regulalized lineaar regression. To minimized the loss function, we need the rugulized para lambda to increase. Both in the case of number of examples of number of features, as it increases, we need bigger lambda for balance to avoid overfitting.

# ## Task 2

# In[8]:


datasize=np.arange(10,800,20)


def l_C(lam,traininputs,traintargets,testinputs,testtargets):
    
    mse_vec_te_mean=[]
    for size in datasize:
        mse_vec_te=[]
        for repeat in range(15):
            
            sample_in_100_row = [np.random.randint(0,998) for i in range(size)]

            xtrain=traininputs.values[sample_in_100_row]
            ttrain=traintargets.values[sample_in_100_row]
            xtest=testinputs.values[sample_in_100_row]
            ttest=testtargets.values[sample_in_100_row]

            lambd = np.diag(np.ones(xtrain.shape[1])*lam)
            w=np.dot(np.dot(np.linalg.inv(lambd+np.dot(xtrain.transpose(),xtrain)),xtrain.transpose()),ttrain)
            mse_te=np.sum(((np.dot(xtest,w)-ttest)**2))/ttest.shape[0]
            mse_vec_te.append(mse_te)
        mse_vec_te_mean.append(np.mean(mse_vec_te))

    return mse_vec_te_mean


# In[9]:


mse_vec_te_mean_lambda04=l_C(4,train_1000_100,trainR_1000_100,test_1000_100,testR_1000_100)
mse_vec_te_mean_lambda18=l_C(18,train_1000_100,trainR_1000_100,test_1000_100,testR_1000_100)
mse_vec_te_mean_lambda28=l_C(28,train_1000_100,trainR_1000_100,test_1000_100,testR_1000_100)


plt.plot(datasize, mse_vec_te_mean_lambda04)
plt.plot(datasize, mse_vec_te_mean_lambda18)
plt.plot(datasize, mse_vec_te_mean_lambda28)

plt.xlabel("range of datasize")
plt.ylabel("mse")
plt.legend(["lambda=4", "lambda=8", "lambda=28"])
plt.show()


# ## Q&A for Task 2
# 
# Q:In your report provide the results/plots and discuss them: What can you observe from the plots regarding the dependence on lambda and the number of samples? Consider both the case of small training set sizes and large training set sizes. How do you explain these variations?
# 
# A: First, as the lambda increase from 4 to 18 to 28, the MSE of train set will increase. Second, as the number of datasize increase from 0 to 800, the MSE will decrease too. 
# In the big data size case, the mse of lambda4 is biger than lambda18, and the mse of lambda28 is the least. Because the lambda 28 is the closest to optimun lambda.
# In the big data size case, the mse of lambda4 is small than lambda18, and the mse of lambda28 is the largest.

# ## Task 3

# In[10]:


#the model selection function
def b_m_s(traininputs,traintargets):
    
    alpha=1
    beta=1
    criterion=1
    
    #data prepare
    xtrain=traininputs.values
    ttrain=traintargets.values
    
    #set up a criterion to justify the optimun alpha and beta in irretation
    while criterion>0.00001:
    
        Sn=np.linalg.inv(np.identity(xtrain.shape[1])*alpha + beta*np.dot(xtrain.transpose(),xtrain))
        Mn=beta*np.dot(Sn,np.dot(xtrain.transpose(),ttrain))
        
        #eigen value of beta.x(transform)*x
        lambd_blr=np.linalg.eigvals(beta*np.dot(xtrain.transpose(),xtrain))

        gamma=sum(lambd_blr/(lambd_blr+alpha))

        alpha_upda = gamma/(Mn**2).sum()
        
        dist=0
        for n in range(xtrain.shape[0]):
            dist += (ttrain[n][0] - (Mn.transpose() * xtrain[n]).sum()) ** 2
        beta_upda=(xtrain.shape[0]-gamma)/dist
    
        #the form of criterion
        criterion = abs(abs(alpha_upda - alpha)/alpha + abs(beta_upda-beta)/beta)
 
        alpha = alpha_upda
        beta = beta_upda
    
    
    return alpha, beta


# In[11]:


#an example for 10-100 data
alpha_10,beta_10=b_m_s(train_100_10,trainR_100_10)


# In[12]:


print ('an example for alpha and beta in 10-100 data')
print ('alpha_10 is',alpha_10)
print ('beta_10 is', beta_10)


# In[13]:


#calculate the lambda and mse for the bayesian leaning regression
def b_l_r(traininputs,traintargets,testinputs,testtargets):
    
    xtrain=traininputs.values
    ttrain=traintargets.values
    xtest=testinputs.values
    ttest=testtargets.values
    
    mse_vec_tr=[]
    mse_vec_te=[]
    
    alpha,beta=b_m_s(traininputs,traintargets)
    
    lam=alpha/beta
    lambd = np.diag(np.ones(xtrain.shape[1])*lam)
    w=np.dot(np.dot(np.linalg.inv(lambd+np.dot(xtrain.transpose(),xtrain)),xtrain.transpose()),ttrain)
    mse_tr=np.sum(((np.dot(xtrain,w)-ttrain)**2))/ttrain.shape[0]
    
    mse_vec_tr.append(mse_tr)
        
    mse_te=np.sum(((np.dot(xtest,w)-ttest)**2))/ttest.shape[0]
    mse_vec_te.append(mse_te)
    
    return lam,mse_vec_te


# In[14]:


lam_10, msete_10 = b_l_r(train_100_10,trainR_100_10,test_100_10,testR_100_10)
lam_100, msete_100 = b_l_r(train_100_100,trainR_100_100,test_100_100,testR_100_100)
lam_1000, msete_1000 = b_l_r(train_1000_100,trainR_1000_100,test_1000_100,testR_1000_100)
lam_f3, msete_f3 = b_l_r(train_f3,trainR_f3,test_f3,testR_f3)
lam_f5, msete_f5 = b_l_r(train_f5,trainR_f5,test_f5,testR_f5)

print ('Bayesian linear Regression Model Selection:')

print ('for 10-100, the lambda is')
print (lam_10)
print ('for 10-100, the test mse is')
print (msete_10)

print ('for 100-100, the lambda is')
print (lam_100)
print ('for 100-100, the test mse is')
print (msete_100)

print ('for 1000-100, the lambda is')
print (lam_1000)
print ('for 1000-100, the test mse is')
print (msete_1000)

print ('for f3, the lambda is')
print (lam_f3)
print ('for f3, the test mse is')
print (msete_f3)

print ('for f5, the lambda is')
print (lam_f5)
print ('for f5, the test mse is')
print (msete_f5)


# ## Q&A for Task 3
# Q:How do the results compare to the best test-set results from part 1 both in terms of the choice of lambda and test set MSE? (Note that our knowledge from part 1 is with hindsight of the test set, so the question is whether model selection recovers a solution which is close to the best in hindsight.) How does the quality depend on the number of examples and features?
# 
# A: The bayesian linear regression give us a little smaller lambda compared to the result in 1. Meanwhile, the mse given in the bayesian learning are mostly larger than that given in the part 1 model.
# By comparing 1000-100, 100-100 and 10-100, we can found out that when the number of examples is much bigger than that of features, the quality is much better. Otherwise, the matrix is not full-mark, which will make it not invertable.

# ## Task 4

# In[15]:


#calculate logevidence and MSE for bayesian and non regularized model
def log_v(traininputs,traintargets,testinputs,testtargets):
    
    #data prepare
    xtrain_raw=traininputs.values
    ttrain=traintargets.values
    xtest_raw=testinputs.values
    ttest=testtargets.values
    
    #initialize empty vector for storage
    alpha_vec=[]
    beta_vec=[]
    log_evidence_list=[]
    mse_vec_te_nonreg=[]
    mse_vec_te_bl=[]
    
    
    #generate different dimensions
    for d in range(10):
#         poly = PolynomialFeatures(degree=d)
        
#         xtrain = poly.fit_transform(xtrain_raw)
#         xtest = poly.fit_transform(xtest_raw)
        
        xtrain=np.power(xtrain_raw[:None],np.arange(0,d+1))
        xtest=np.power(xtest_raw[:None],np.arange(0,d+1))
        
        
        
        #for bayesian linear regression
        alpha=8
        beta=8
        criterion=1


        #calculate alpha and beta
        while criterion>0.00001:
    
            Sn=np.linalg.inv(np.identity(xtrain.shape[1])*alpha + beta*np.dot(xtrain.transpose(),xtrain))
            Mn=beta*np.dot(Sn,np.dot(xtrain.transpose(),ttrain))
            

            lambd_blr=np.linalg.eigvals(beta*np.dot(xtrain.transpose(),xtrain))

            gamma=sum(lambd_blr/(lambd_blr+alpha))

            alpha_upda = gamma/(Mn**2).sum()

            dist=0
            for n in range(xtrain.shape[0]):
                dist += (ttrain[n][0] - (Mn.transpose() * xtrain[n]).sum()) ** 2
            beta_upda=(xtrain.shape[0]-gamma)/dist

            criterion = abs(alpha_upda - alpha)/alpha + abs(beta_upda-beta)/beta
            alpha = alpha_upda
            beta = beta_upda

        #return alpha,beta
        alpha_vec.append(alpha)
        beta_vec.append(beta)
        
        #caculate A
        A=np.diag(np.ones(xtrain.shape[1])*alpha)+beta*np.dot(xtrain.transpose(),xtrain)
        

        #caculate E(mn)
        dist1=0
        for n in range(xtrain.shape[0]):
            dist1 += ((Mn.transpose() * xtrain[n]).sum()-ttrain[n][0] ) ** 2
        EMN=beta*1/2*dist1+beta*1/2*(np.sum(Mn**2))


        #calculate log evidence
        log_evidence=xtrain.shape[1]/2*np.log(alpha)+xtrain.shape[0]/2*np.log(beta)-EMN-1/2*np.log(np.linalg.det(A))-xtrain.shape[0]/2*np.log(2*math.pi)

        log_evidence_list.append(log_evidence)
        
        #calculate mse for bl
        '''
        dist3=0
        for n in range(ttest.shape[0]):      
              dist3 += (ttest[n][0]-Mn[0]) ** 2 #I try to use the Mn but the demension doesn't fit here
        mse_te_bl=dist3/ttest.shape[0]        
        mse_vec_te_bl.append(mse_te_bl)
        '''
        
        lam=alpha/beta
        lambd = np.diag(np.ones(xtrain.shape[1])*lam)
        w=np.dot(np.dot(np.linalg.inv(lambd+np.dot(xtrain.transpose(),xtrain)),xtrain.transpose()),ttrain)
        mse_te_bl=np.sum(((np.dot(xtrain,w)-ttrain)**2))/ttrain.shape[0]
        mse_vec_te_bl.append(mse_te_bl)

       
    
        #for non regularized linear regression
        
        #it equals to set lambd_nonreg = np.diag(np.ones(xtrain.shape[1])*0)
        w_nonreg=np.dot(np.dot(np.linalg.inv(np.dot(xtrain.transpose(),xtrain)),xtrain.transpose()),ttrain)
        
        #calculate mse for non reg
        mse_te_nonreg=np.sum(((np.dot(xtest,w_nonreg)-ttest)**2))/ttest.shape[0]
        mse_vec_te_nonreg.append(mse_te_nonreg)
        
    
    
    return alpha_vec,beta_vec,log_evidence_list,mse_vec_te_bl,mse_vec_te_nonreg


# In[16]:


#input data f3 and f5
alphaf3,betaf3,log_evidencef3,mse_vec_te_blf3,mse_vec_te_nonregf3=log_v(train_f3,trainR_f3,test_f3,testR_f3)
alphaf5,betaf5,log_evidencef5,mse_vec_te_blf5,mse_vec_te_nonregf5=log_v(train_f5,trainR_f5,test_f5,testR_f5)


# In[18]:


#plot alpha and beta on f3 and f5
plt.title("alpha on d")
plt.plot(range(10), alphaf3)
plt.xlabel("range of complexity")
plt.ylabel("alpha for bayesian leaning")

plt.plot(range(10), alphaf5)
plt.xlabel("range of complexity")
plt.ylabel("alpha for bayesian leaning")
plt.legend("f3","f5")
plt.show()

plt.title("beta on d")
plt.plot(range(10), betaf3)
plt.xlabel("range of complexity")
plt.ylabel("beta for bayesian leaning")


plt.plot(range(10), betaf5)
plt.xlabel("range of complexity")
plt.ylabel("beta for bayesian leaning")
plt.legend("f3","f5")
plt.show()


# In[17]:


#plot logevidence mse(bl) and mse(non regularized) on f3 data
plt.title("logevidence on d for f3")
plt.plot(range(10), log_evidencef3)
plt.xlabel("range of dimension")
plt.ylabel("log_evidence")
plt.show()

plt.title("mse for bl on d for f3")
plt.plot(range(10), mse_vec_te_blf3)
plt.xlabel("range of dimension")
plt.ylabel("mse for bl")
plt.show()

plt.title("mse for non regularized on d for f3")
plt.plot(range(10), mse_vec_te_nonregf3)
plt.xlabel("range of dimension")
plt.ylabel("mse for  nonreg")
plt.show()


plt.title("logevidence on d for f5")
plt.plot(range(10), log_evidencef5)
plt.xlabel("range of dimension")
plt.ylabel("log_evidence")
plt.show()

plt.title("mse for bl on d for f5")
plt.plot(range(10), mse_vec_te_blf5)
plt.xlabel("range of dimension")
plt.ylabel("mse for bl")
plt.show()

plt.title("mse for non regularized on d for f5")
plt.plot(range(10), mse_vec_te_nonregf5)
plt.xlabel("range of dimension")
plt.ylabel("mse for  nonreg")
plt.show()


# ## Q&A for Task 4
# 
# Q:Can the evidence be used to successfully select alpha, beta and d for the Bayesian method? How does the non-regularized model fare in these runs? (Note that evidence is only relevant for the Bayesian method and one would need some other method to select d in this model)
# 
# A: The evidence can be used to successfully select alpha, beta, and for d for the Bayesian method. For each d, we show the responding alpha and beta. And more importantly, we can see that it gives the optimum d, at about d=3 with the maximum log evidence. In the Bayesian Learning model, when we try to calculate the MSE of test data, we can also see that it firstly decrease with the number of features,and then it has a increases (for f5 but not for f3 in the graph). This is because as the dimension increases, the variance will contribute more than bias, which leads to potential U curve MSE on test deta. In the non-regularized model, the model keep decreasing when the complexity is big. This is because when it is not regularized, the bias of the model will decrase as more demension we have in the model.
