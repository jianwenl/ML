
# coding: utf-8

# # Project 1 on Machine Learning-Unigram model

# In[1]:


#import library
import numpy as np
import matplotlib.pyplot as plt


#load the data from dropbox and split the data
def f2l(file_name):
    with open(file_name, 'r') as f:
        list = [word for line in f for word in line.split()]
        f.close()
    return list

train_data=f2l("training_data.txt")
test_data=f2l("test_data.txt")


# create an union vocabulary
all_data=train_data+test_data
VocaUnite=set(all_data)
K_all = len(VocaUnite)
print (K_all)


# In[2]:


#get different subsets of train set
N=len(train_data)

#use a list to represent the division
trainsizesplit = [int(N/128), int(N/64), int(N/16), int(N/4), int(N)]


# split the dataset
train_data_subsets = [train_data[0:size] for size in trainsizesplit]


#train_data_subsets[0] = train_data[0:int(N/128)]
#train_data_subsets[1] = train_data[0:int(N/64)]
#train_data_subsets[2] = train_data[0:int(N/16)]
#train_data_subsets[3] = train_data[0:int(N/4)]
#train_data_subsets[4] = train_data[0:int(N)]

print (trainsizesplit)
print (N)


# In[3]:


#create the dictionary and count on each words appears in train data
def count(training_data,vocabulary):
    countU = {}.fromkeys(vocabulary,0)
    for k in training_data:
        countU[k] += 1
    return countU

#length of training data
def leng_of_train(training_data):
    leng=len(training_data)
    return leng

#define the Perplexity function
def pp(test_data,countU,NTR):
    
    #the length of test data
    NTE=len(test_data) 
      
    # Dir priors
    alpha=2
    
    #PP value for three methods
    #pp value of ML
    ML_logsum=0
    for k in test_data:    
        
        mk=countU[k]
        #we construct a=0.045 for the reason of weakness of ML and scale on data
        if mk==0:
            DML=0.045
        else:
            DML=mk
        Probs_ML=DML/(NTE+NTR)
        ML_logsum +=np.log(Probs_ML)#I try to use numpy.sum but the test pp is much less than train data, which I cannot figure out
    PPVML=np.exp(((-1)/NTE)* ML_logsum)
        
    #pp value of MAP
    MAP_logsum=0
    for k in test_data: 
        mk=countU[k]
        Probs_MAP=(mk+alpha-1)/((NTE+NTR)+alpha*K_all-K_all)
        MAP_logsum += np.log(Probs_MAP)
    PPVMAP=np.exp(((-1)/NTE)* MAP_logsum)
        
    #pp value of PR
    PR_logsum=0
    for k in test_data:
        mk=countU[k]
        Probs_PR=(mk+alpha)/((NTE+NTR)+alpha*K_all)
        PR_logsum += np.log(Probs_PR)
    PPVPR=np.exp(((-1)/NTE)*PR_logsum)
    
    
    return [PPVML, PPVMAP, PPVPR]


# In[4]:


#train the model with 5 data sets

countU1=count(train_data_subsets[0],VocaUnite)
NTR1=len(train_data_subsets[0])
test_11=pp(test_data,countU1,NTR1)
train_11=pp(train_data_subsets[0],countU1,NTR1)

countU2=count(train_data_subsets[1],VocaUnite)
NTR2=len(train_data_subsets[1])
test_21=pp(test_data,countU2,NTR2)
train_21=pp(train_data_subsets[1],countU2,NTR2)

countU3=count(train_data_subsets[2],VocaUnite)
NTR3=len(train_data_subsets[2])
test_31=pp(test_data,countU3,NTR3)
train_31=pp(train_data_subsets[2],countU3,NTR3)

countU4=count(train_data_subsets[3],VocaUnite)
NTR4=len(train_data_subsets[3])
test_41=pp(test_data,countU4,NTR4)
train_41=pp(train_data_subsets[3],countU4,NTR4)

countU5=count(train_data_subsets[4],VocaUnite)
NTR5=len(train_data_subsets[4])
test_51=pp(test_data,countU5,NTR5)
train_51=pp(train_data_subsets[4],countU5,NTR5)


#test on different size of trainning data using 3 methods
PPMLTE=[test_11[0],test_21[0],test_31[0],test_41[0],test_51[0]]
PPMAPTE=[test_11[1],test_21[1],test_31[1],test_41[1],test_51[1]]
PPPRTE=[test_11[2],test_21[2],test_31[2],test_41[2],test_51[2]]

#train on different size of trainning data using 3 methods
PPMLTR=[train_11[0],train_21[0],train_31[0],train_41[0],train_51[0]]
PPMAPTR=[train_11[1],train_21[1],train_31[1],train_41[1],train_51[1]]
PPPRTR=[train_11[2],train_21[2],train_31[2],train_41[2],train_51[2]]


#
print (PPMLTE)
print (PPMAPTE)
print (PPPRTE)

print (PPMLTR)
print (PPMAPTR)
print (PPPRTR)


# In[5]:


#plot the figure
plt.plot([0, 1, 2, 3, 4],PPMLTE,[0, 1, 2, 3, 4],PPMAPTE,[0, 1, 2, 3, 4],PPPRTE,[0, 1, 2, 3, 4],
         PPMLTR,[0, 1, 2, 3, 4],PPMAPTR,[0, 1, 2, 3, 4],PPPRTR)
plt.title("perplexity on Training Sets")
plt.xlabel("Training sets of different sizes")
plt.xticks([0, 1, 2, 3, 4], ["subset1","subset2","subset3","subset4","subset5"])
plt.ylabel("Perplexity")
plt.legend(["MLtest","MAPtest","PRtest","MLtrain","MAPtrain","PRtain"])
plt.show()


# ## Task 1
# 
# Q1:What happens to the test set perplexities of the different methods with respect to each other as the training set size increases? Please explain why this occurs. 
# 
# A1:As the training set size increase, we can see that the perplexity of test sets basically decreases on all the three methods both on training and teseting dataset (in fact both in the test data case and in the training data case the perplexity will also decrease based on the numerical result).
# 
# Another observation I want to discuss is that the perplexities of test data set is bigger compared to that of train dataset. As the dataset size increases, we can observe from the figure that the perplexities will decrease and converge.
# 
# The reason for the decrease is that a biger training sets leads to the smaller bias for the model. This relation between the perplexity and the dataset size does not depend on which method we use.
# 
# Q2:What is the obvious shortcoming of the maximum likelihood estimate for a unigram model? How do the other two approaches address this issue?
# 
# A2: In the maximum likelihood estimate we can found out that it can not hold the special case in which some vacobulary do not appear at all. In this scenario the log of probability zero will leads to an infinite number in the expression. This break dowm the ML method. To avoid this, I use an alternative Constance to replace zero in case zero appears.
# 
# Q3: For the full training set, how sensitive do you think the test set perplexity will be to small changes in alpha? why?
# 
# A3: The test set perplexity will not change too much in alpha. The counting number of training data will overcome the priors (pseudo count) in the possibility expression. Because in the Baysesian update, the effect weighted by data has far surpassed that from alpha, which is the original priors.
# 
# 

# In[6]:


#Task 2

Ntask2=int(N/128)

#create dictionary
countTask2={}.fromkeys(VocaUnite,0)

#create a count on training data
for k in train_data_subsets[0]:
    countTask2[k]+=1
       
#the log evidence caculation

def le_calculate():        
    #for numerator
    #le_list=list()
    le_list=list()
    def le_logsum_num():
        le_logsum_numx=0
        #for every word in the counting of trainng set
        for k in countTask2:
            mk=countTask2[k]
            #count on every time the word appears 
            m=0
            while m <= (mk-1):
                le_logsum_numx+=np.log(alpha+m)#cancel the factorial
                m+=1
        return le_logsum_numx

    #for denumerator
    def le_logsum_denum():
        le_logsum_denumx=0
        n=0
        while n <= Ntask2-1:
            le_logsum_denumx+=np.log(alpha*K_all+n)#cancel the factorial
            n+=1
        return le_logsum_denumx
    
    #main loop on alpha
    alpha=0
    while alpha <10:
    #for alpha in alphaS:
        le_logsum=0
        num=le_logsum_num()
        denum=le_logsum_denum()
        le_logsum=num-denum
        #le_list.append(le_logsum)
        le_list.insert(alpha-1,le_logsum)
        alpha+=1
    return le_list
    
    
#the PP calculation

def pp_calculate():    
    #main loop on alpha
    PP_list=list()
    alpha=1
    while alpha <11:
    #for alpha in alphaS:
        #calculate the logsum of PP
        p_logsum=0
        for k in test_data:
            m=countTask2[k]
            #we use the bayesian method-predictive distribution probability-to calculate its PP value.ps: the MAP will have a inf value
            p_logsum+=np.log((m+alpha)/(Ntask2+alpha*K_all))
        PP_list.insert(alpha-1,np.exp(-1/N*p_logsum))
        alpha+=1
    return PP_list


le=le_calculate()
PP=pp_calculate()
print (le)
print (PP)


# In[7]:


#plot the figure of log evidence
plt.plot([1,2,3,4,5,6,7,8,9,10],le)
plt.title("log evidence on alpha")
plt.xticks([1,2,3,4,5,6,7,8,9,10], ["1","2","3","4","5","6","7","8","9","10"])
plt.xlabel("alpha value")
plt.ylabel("log evidence")
plt.show()


#plot the figure of PP
plt.plot([1,2,3,4,5,6,7,8,9,10],PP)
plt.title("PP on alpha of PR")
plt.xticks([1,2,3,4,5,6,7,8,9,10], ["1","2","3","4","5","6","7","8","9","10"])
plt.xlabel("alpha value")
plt.ylabel("PP")
plt.show()


# ## Task 2
# 
# Q:Is maximizing the evidence function a good method for model selection on this dataset? 
# 
# A: As shown in the result, the data of log evidence is Concave depending on the alpha. It has a maximum point. On the other hand, the perplexity is Convex depending on the alpha. It has a minimum point. Maximizing the evidence function is a good method for model selection on this dataset. Because we can see that the point of alpha that maximizing the evidence function is the same point that minimizing Perplexity. In addition, the trend of curve on two figures are just the opposite which makes the evidence function a good method more convincing.
# 

# In[8]:


#task 3

#load the data into a list
def f2l(file_name):
    with open(file_name, 'r') as f:
        list = [word for line in f for word in line.split()]
        f.close()
    return list

train_data3=f2l("pg121.txt.clean")
test_data31=f2l("pg141.txt.clean")
test_data32=f2l("pg1400.txt.clean")


# create an union vocabulary
alldata3=train_data3+test_data31+test_data32
VocaUnite3=set(alldata3)

countU31=count(train_data3,VocaUnite3)
NTR31=len(train_data3)
test_31=pp(test_data31,countU31,NTR31)

countU32=count(train_data3,VocaUnite3)
NTR32=len(train_data3)
test_32=pp(test_data32,countU32,NTR32)


print (test_31)
print (test_32)


# ## Task3
# Q:Was the model successful in this classification task? Please report and discuss these results.
# 
# A:The file pg141 has the smaller PP value, which means that it has the same author of that of pg121. Therefore, the model can be sucessful in author identification.
