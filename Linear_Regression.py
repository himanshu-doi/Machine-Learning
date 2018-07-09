
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os as os


# ### Reading dataset

# In[2]:


os.chdir('/home/himanshu/wine_data')
data_red = pd.read_csv('winequality-red.csv', sep = ';')


# In[3]:


data_red.shape


# In[4]:


data_red.head()


# In[5]:


data_red.describe()


# In[6]:


train_y = data_red.quality.values


# In[7]:


train_y = train_y.reshape(1,-1)


# In[8]:


train_y.shape


# In[9]:


train_x = data_red.iloc[:,0:11].values


# In[10]:


train_x.shape


# In[11]:


#Splitting dataset into training and cross-validation sets
from sklearn.cross_validation import train_test_split

X_train, X_cv, Y_train, Y_cv = train_test_split(train_x, data_red.quality, test_size = 0.2, random_state = 56)


# In[12]:


Y_train = Y_train.reshape(1,-1)
Y_cv = Y_cv.reshape(1,-1)


# In[13]:


Y_train.shape


# ### Linear Regression Model

# In[14]:


features = data_red.columns[:-1]


# In[15]:


## no. of data points
m = len(train_x)       

## initialting weight matrices
t0 = 1

W = np.zeros(train_x.shape[1]).reshape(1,-1)
W.shape


# In[41]:


def cost_func(x, y, W):

    ## cost function expression
    J = np.sum((W.dot(x.T) - y)**2/(2*m)) 

    return J


# In[17]:


initial_cost = cost_func(train_x, train_y)
print 'initial_cost = %f' %(initial_cost)


# In[19]:


def Gradient_descent(x, y, alpha, max_iter):
    
    W = np.zeros(x.shape[1]).reshape(1,-1)
    cost_i = [0]*max_iter
    m = len(x)
    t0 = 1
       
    for i in range(max_iter):
        h = t0 + W.dot(x.T)    ## hypothesis func
        
        loss = h - y     ## training loss
        
        ## gradient calculation
        grad0 = loss/m
        grad1 = np.dot(loss,x)/m
        
        ## updating weights
        t0 = t0 - alpha*grad0
        W = W - alpha*grad1 
        
        ## cost function  
        #cost = cost_func(x, y)
        cost = np.sum((W.dot(x.T) - y)**2/(2*m))
        
        cost_i[i] = cost
        #print 'training_cost[i] = %f' %(cost_i[i])
        
    plt.plot(range(max_iter), cost_i)
    plt.ylabel('Training loss')
    plt.xlabel('Iteration')
    plt.show()
    
        
    return h, t0, W,cost

    


# In[20]:


## Training Cycle
Y_pred_t, bias_t, weights_t, cost_t = Gradient_descent(X_train,Y_train, alpha=10e-5, max_iter=1000)


# In[42]:


## Test Cycle

Y_pred = weights_t.dot(X_cv.T) + bias_t.reshape(-1,1)[-1]
test_cost = cost_func(X_cv,Y_cv,weights_t)
print test_cost


# ### Model Evaluation

# In[43]:


## Model Evaluation - Root mean sq. error
def rmse(Y, Y_pred):
    rmse = np.sqrt(np.sum((Y - Y_pred) ** 2)/len(Y))
    return 'rmse = %f' %(rmse)

## Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = np.sum((Y - mean_y) ** 2)
    ss_res = np.sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return 'r2_score = %f' %(r2)


print(rmse(Y_cv, Y_pred))
print(r2_score(Y_cv, Y_pred))

