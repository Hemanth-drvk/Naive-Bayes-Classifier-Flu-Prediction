#!/usr/bin/env python
# coding: utf-8

# # Naïve Bayes classifier to determine if a person has the flu 

# ### Classifier with labels

#  - Yes = 1, No = 0<br>
#  - No headache = 0, Mild = 1, Strong = 2<br>

# In[1]:


import numpy as np
from collections import Counter, defaultdict
 
def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob
 
def naive_bayes(training, outcome, new_sample):
    classes     = np.unique(outcome)
    rows, cols  = np.shape(training)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)
 
    class_probabilities = occurrences(outcome)
 
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])
 
    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])
 
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods[cls][i]
             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
             else:
                 class_probability *= 0
             results[cls] = class_probability
    print(results)
 
if __name__ == "__main__":
    training   = np.asarray(((1,0,1,1),(1,1,0,0),(1,0,2,1),(0,1,1,1),(0,0,0,0),(0,1,2,1),(0,1,2,0),(1,1,1,1)));
    outcome    = np.asarray((0,1,1,1,0,1,0,1))
    new_sample = np.asarray((1,0,1,0))
    naive_bayes(training, outcome, new_sample)


# ##### Output: 
# #### 0 rounded to 4 decimals is 0.0185
# #### 1 rounded to 4 decimals is 0.0006

# ### Classifier with words surrounded by quotes

# In[2]:


import numpy as np
from collections import Counter, defaultdict
 
def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob
 
def naive_bayes(training, outcome, new_sample):
    classes     = np.unique(outcome)
    rows, cols  = np.shape(training)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)
 
    class_probabilities = occurrences(outcome)
 
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])
 
    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])
 
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods[cls][i]
             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
             else:
                 class_probability *= 0
             results[cls] = class_probability
    print(results)
 
if __name__ == "__main__":
    training   = np.asarray((('Y','N','Mild','Y'),('Y','Y','N','N'),('Y','N','Strong','Y'),('N','Y','Mild','Y'),('N','N','No','N'),('N','Y','Strong','Y'),('N','Y','Strong','N'),('Y','Y','Mild','Y')));
    outcome    = np.asarray(('N','Y','Y','Y','N','Y','N','Y'))
    new_sample = np.asarray(('Y','N','Mild','N'))
    naive_bayes(training, outcome, new_sample)


# ##### Output: 
# #### 'No'  rounded to 4 decimals is 0.0185
# #### 'Yes'  rounded to 4 decimals is 0.0006

# Rinda Digamarthi(157742d)

# In[ ]:




