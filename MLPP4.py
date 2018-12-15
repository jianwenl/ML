
# coding: utf-8

# # PP4 LDA and Classification

# ## Task 1

# In[1]:


import numpy as np
import scipy as sci
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


def train_data(filename):
    with open("/Users/jianwenliu/MLPP4/20newsgroups/" + filename, 'r') as file:
        data = file.read().split(" ")
        data.remove("")
    return data


# In[3]:


a=train_data(str(1))
a


# In[4]:


#initial the topic and docment paras
K=20
D=200
Niters=500
alpha= 5/K
beta=0.01


# In[5]:


data = np.array([])
d = np.array([],dtype = int)
z = np.array([],dtype = int)
data = np.array([])
d = np.array([],dtype = int)
z = np.array([],dtype = int)
for i in range(1,D+1):
    data_i = train_data(str(i))
    length_i = len(data_i)
    
    d_i = i*np.ones([1,length_i], dtype = int)
    
    z_i = np.random.randint(1, K+1, [1,length_i])
    
    data = np.append(data, data_i)
    #array of document indices d(n)
    d = np.append(d, d_i)
    #array of initial topic indices z(n)
    z = np.append(z, z_i)

N_words = len(data)
N_words


# In[6]:


data


# In[7]:


#the last document array
d_i


# In[8]:


d.shape


# In[9]:


collection=set(data)
Voc = list(collection)
Voc.sort()
V = len(Voc)


# In[10]:


V


# In[11]:


# Initialize w(n)

#initial a empty array for w
w = np.zeros(N_words, dtype = int)
word_index = dict.fromkeys(Voc,0)

for i in range(V):
    word_index[Voc[i]] = i+1


for i in range(N_words):
    w[i] = word_index[data[i]]
    


# In[12]:


w


# In[13]:


# Initialized the distribution pi
pi = np.random.permutation(range(0,N_words))


# In[14]:


pi


# In[15]:


# for cd and ct
#initial cd and ct
# Compute Cd and Ct
Cd = np.zeros([D,K])
for i in range(D):
    for j in range(K):
        Cd[i,j] = sum((d == i+1) * (z == j+1))
        
Ct = np.zeros([K,V])
for i in range(K):
    for j in range(V):
        Ct[i,j] = sum((z == i+1) * (w == j+1))
        
# Initialize Possibilities
P = np.zeros(K)


# In[16]:


# Gibbs sampling
for iter in range(Niters):
    for n in range(N_words):
        word = w[pi[n]]
        topic = z[pi[n]]
        doc = d[pi[n]]
        Cd[doc-1,topic-1] = Cd[doc-1,topic-1] - 1
        Ct[topic-1,word-1] = Ct[topic-1,word-1] - 1
        for k in range(K):
            P[k] = (Ct[k,word-1] + beta)/(V*beta + sum(Ct[k,:])) * (Cd[doc-1,k] + alpha)/(K*alpha + sum(Cd[doc-1,:]))
        P = P/sum(P)
        topic = np.random.choice(range(1,K+1),p = P)
        z[pi[n]] = topic
        Cd[doc-1,topic-1] = Cd[doc-1,topic-1] + 1
        Ct[topic-1,word-1] = Ct[topic-1,word-1] + 1


# In[17]:


#we can change the paras here to get the top number words in each topic
M_freq_words = np.zeros([K,5] ,dtype = object)
for i in range(K):
    M_freq = np.argsort(Ct[i])[-5:][::-1]
    for j in range(5):
        M_freq_words[i,j] = Voc[M_freq[j]]
print (M_freq_words)
df = pd.DataFrame(M_freq_words)
df.to_csv("topicwords.csv",header = False, index = False)


# ## Q&A for Task 1
# 
# A: The top 5 frequenst words in each topic is shown as above.

# ## Task 2

# In[18]:


#data pre
X_topics = np.zeros([D,K])
for i in range(D):
    for j in range(K):
        X_topics[i,j] = (Cd[i,j] + alpha)/(K*alpha + np.sum(Cd[i,:]))

X_bagofwords = np.zeros([D,V])
for i in range(D):
    for j in range(V):
        X_bagofwords[i,j] = (sum((d == i+1) * (data == Voc[j])) + 0.0) / sum(d == i+1)

Y = pd.read_csv("/Users/jianwenliu/MLPP4/20newsgroups/index.csv", header = None).iloc[:,[1]].values


# In[19]:


#prepare for the train/test data
def data_split(X,Y):
    N = X.shape[0]
    
    index = list(range(N))
    np.random.shuffle(index) # Randomly shuffle the indices
    
    train_index = index[0:int(2*N/3)]
    test_index = index[int(2*N/3):N]
    
    Xtrain = X[train_index]
    Ytrain = Y[train_index]
    Xtest = X[test_index]
    Ytest = Y[test_index]
    return Xtrain, Ytrain, Xtest, Ytest


# In[20]:


#make use of the function in last pp
def b_log(Xtrain, Ytrain, Xtest, Ytest):
    
    N = Xtrain.shape[0]
    onecol = np.array([[1]]*N)
    Xtrain = np.concatenate((onecol, Xtrain), axis = 1) # Add one column of ones to the features
    M = Xtrain.shape[1]

    phi = Xtrain
    t = Ytrain
    alpha = 0.01
    w = np.array([[0]]*M)
    criterion = 1
    n = 1


    #train to get W
    while criterion > 10**-3 and n < 100:
        w_pri = w

        a = phi.dot(w_pri)
        y = 1.0/(1 + np.exp(-a))
        r = y*(1-y)
        R = np.diag(r.ravel())
        I = np.eye(M)
        A = alpha * I + phi.transpose().dot(R.dot(phi))
        B = phi.transpose().dot(y - t) + alpha * w_pri
        w_upd = w_pri - np.linalg.inv(A).dot(B)

        criterion = np.linalg.norm(w_upd - w_pri) / np.linalg.norm(w_pri)
        w = w_upd
        n += 1 
    
    #for SN
    a = phi.dot(w)
    y = 1/(1 + np.exp(-a))
    SN_inverse = alpha * I
    for n in range(N):
        SN_inverse += y[n]*(1-y[n]) * np.outer(phi[n],phi[n])
    SN = np.linalg.inv(SN_inverse)

    # test
    onecol = np.array([[1]]*Xtest.shape[0])
    Xtest = np.concatenate((onecol, Xtest), axis = 1)
    phi_test = Xtest
    perf = []
    for n in range(Xtest.shape[0]):
        mu_a = phi_test[n].dot(w)
        sigma_a2 = phi_test[n].transpose().dot(SN.dot(phi_test[n]))
        kap = (1 + np.pi * sigma_a2 / 8) ** (-1.0/2)
        p = 1.0/(1 + np.exp( - kap * mu_a))
        if p >= 0.5:
            perf.append(int(Ytest[n] == 1))
        else:
            perf.append(int(Ytest[n] == 0))
    acr = (sum(perf))/len(perf)
    
    return acr


# In[21]:



#initial a list to save s
ratio= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
acr_t = np.zeros([30,len(ratio)])
acr_b = np.zeros([30,len(ratio)])

for k in range(30):
    Xtrain_t, Ytrain_t, Xtest_t, Ytest_t = data_split(X_topics,Y)
    Xtrain_b, Ytrain_b, Xtest_b, Ytest_b = data_split(X_bagofwords,Y)
    j = 0
    for r in ratio:
        acr_t[k,j] = b_log(Xtrain_t[0:int(r*D)], Ytrain_t[0:int(r*D)], Xtest_t, Ytest_t)
        acr_b[k,j] = b_log(Xtrain_b[0:int(r*D)], Ytrain_b[0:int(r*D)], Xtest_b, Ytest_b)        
        j += 1

acr_t_mean = np.mean(acr_t, axis = 0)
acr_t_std = np.std(acr_t, axis = 0)
acr_b_mean = np.mean(acr_b, axis = 0)
acr_b_std = np.std(acr_b, axis = 0)


# In[22]:


fig = plt.gcf()
fig.set_size_inches(12, 8)
matplotlib.rcParams.update({'font.size': 15})
plt.errorbar(ratio, acr_t_mean, yerr = acr_t_std, fmt = 'g-*')
plt.errorbar(ratio, acr_b_mean, yerr = acr_b_std, fmt = 'r-^')
plt.xlabel('train data size')
plt.ylabel('accuracy performance')
plt.title('LDA vs Bag-of-words')
plt.legend(['LDA','Bag-of-words'], loc = 4)


# ## Q&A for Task 2:
# 
# Q: Plot the learning curve performance of the logistic regression algorithm on the two representations.Then discuss your observations on the results obtained.
# 
# A: We can find out that the Bag-of-words is more accurate than the LDA when we increase the data size for trainning data. The LDA is little better to bag of words when training data size is small at first.
