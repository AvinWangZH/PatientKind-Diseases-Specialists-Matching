import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
import json
import pickle
import logging


with open('training_data_dict.json', 'r') as f:
    training_data_dict = json.load(f)
    
with open('positive_set.p', 'rb') as f:
    positive_set = pickle.load(f)
    
with open('negative_set.p', 'rb') as f:
    negative_set = pickle.load(f)

logging.basicConfig(level='INFO')
np.random.seed(10)

positive_random_row_num = np.random.choice(positive_set.shape[0], positive_set.shape[0], replace=False)
positive_val_row_num = positive_random_row_num[:int(positive_set.shape[0]/2)]
positive_test_row_num = positive_random_row_num[int(positive_set.shape[0]/2):]

negative_random_row_num = np.random.choice(negative_set.shape[0], negative_set.shape[0], replace=False)
negative_val_row_num = negative_random_row_num[:int(positive_set.shape[0]/2)]
negative_test_row_num = negative_random_row_num[int(positive_set.shape[0]/2):int(positive_set.shape[0])]
negative_training_row_num = negative_random_row_num[int(positive_set.shape[0]):]

num_features = 8
X_positive_val = [] 
X_negative_val = [] 
X_positive_test = [] 
X_negative_test = [] 
X_train = []

count = 1
for row_num in positive_val_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_val.append(list(np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 4:]), axis=0)))
        count += 1
    if count == 1000:
        break
X_positive_val = np.array(X_positive_val)
logging.info('X_positive_val has been finished')

count = 1
for row_num in negative_val_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_val.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        count += 1
    if count == 1000:
        break      
X_negative_val = np.array(X_negative_val)
logging.info('X_negative_val has been finished')

  
count = 1  
for row_num in positive_test_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_test.append(list(np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 4:]), axis=0)))
        count += 1
    if count == 1000:
        break       
X_positive_test = np.array(X_positive_test)
logging.info('X_positive_test has been finished')

count = 1  
for row_num in negative_test_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_test.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        count += 1
    if count == 1000:
        break     
X_negative_test = np.array(X_negative_test)
logging.info('X_negative_test has been finished')

for row_num in negative_training_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_train.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
X_train = np.array(X_train)
logging.info('X_train has been finished')


# generate training set
n_samples = X_positive_val.shape[0] + X_negative_val.shape[0] + X_positive_test.shape[0] + X_negative_test.shape[0] + X_train.shape[0]
X = np.vstack((X_positive_val, X_negative_val))

# generate labels
a = list(np.ones(X_positive_val.shape[0]))
b = list(np.zeros(X_negative_val.shape[0]))
a.extend(b)
labels = np.array(a)


clf = SVR(C=0.5, epsilon=0.2)
clf.fit(X, a) 

#count_right = 0
#count_mim = 0
#for omim_id in training_data_dict:
    #test_set = []
    #expert_index = []
    #count = 0
    #for author in training_data_dict[omim_id]:
        #a = np.array(training_data_dict[omim_id][author][0])
        #if not np.any(np.isnan(a)):
            #test_set.append(training_data_dict[omim_id][author][0]) 
            #if training_data_dict[omim_id][author][1] == 1:
                #expert_index.append(count)
            #count += 1
    
    #if test_set != []:
        #count_mim += 1
        #test_set = np.array(test_set)
        #prediction = clf.predict(test_set)
        #rank = prediction.argsort()
        #for index in expert_index:
            #if rank[index]/len(training_data_dict[omim_id]) < 0.1:
                #print(omim_id, index, rank[index], len(training_data_dict[omim_id]), rank[index]/len(training_data_dict[omim_id]))
                #count_right += 1
                
                
#print(count_right/count_mim)
            

pos_1000 = clf.predict(X_positive_test)
neg_1000 = clf.predict(X_negative_test)
left = clf.predict(X_train)

f, axarr = plt.subplots(2, 2)

axarr[0, 0].boxplot(pos_1000)
axarr[0, 0].set_title('positive 1000')
axarr[0, 1].boxplot(neg_1000)
axarr[0, 1].set_title('negative 1000')
axarr[1, 0].boxplot(left)
axarr[1, 0].set_title('left')

plt.show()
