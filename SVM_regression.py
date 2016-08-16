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
np.random.seed(2)

positive_random_row_num = np.random.choice(positive_set.shape[0], positive_set.shape[0], replace=False)
positive_val_row_num = positive_random_row_num[:int(positive_set.shape[0]/2)]
positive_test_row_num = positive_random_row_num[int(positive_set.shape[0]/2):]

negative_random_row_num = np.random.choice(negative_set.shape[0], negative_set.shape[0], replace=False)
negative_val_row_num = negative_random_row_num[:int(positive_set.shape[0]/2)]
negative_test_row_num = negative_random_row_num[int(positive_set.shape[0]/2):int(positive_set.shape[0])]
negative_training_row_num = negative_random_row_num[int(positive_set.shape[0]):]

num_features = 8
X_positive_val = [] #np.zeros((1, num_features))
X_negative_val = [] #np.zeros((1, num_features))
X_positive_test = [] #np.zeros((1, num_features))
X_negative_test = [] #np.zeros((1, num_features))
X_train = []

count = 1
for row_num in positive_val_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_val.append(list(np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 4:]), axis=0)))
        #X_positive_val = np.vstack((X_positive_val, np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 4:]), axis=0)))
        count += 1
    if count == 1000:
        break
X_positive_val = np.array(X_positive_val)
logging.info('X_positive_val has been finished')

count = 1
for row_num in negative_val_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_val.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        #X_negative_val = np.vstack((X_negative_val, np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        count += 1
    if count == 1000:
        break      
X_negative_val = np.array(X_negative_val)
logging.info('X_negative_val has been finished')

  
count = 1  
for row_num in positive_test_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_test.append(list(np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 4:]), axis=0)))
        #X_positive_test = np.vstack((X_positive_test, np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 4:]), axis=0)))
        count += 1
    if count == 1000:
        break       
X_positive_test = np.array(X_positive_test)
logging.info('X_positive_test has been finished')

count = 1  
for row_num in negative_test_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_test.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        #X_negative_test = np.vstack((X_negative_test, np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        count += 1
    if count == 1000:
        break     
X_negative_test = np.array(X_negative_test)
logging.info('X_negative_test has been finished')

for row_num in negative_training_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_train.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        #X_train.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
X_train = np.array(X_train)
logging.info('X_train has been finished')


# generate ring with inner box
n_samples = X_positive_val.shape[0] + X_negative_val.shape[0] + X_positive_test.shape[0] + X_negative_test.shape[0] + X_train.shape[0]
X = np.vstack((X_positive_val, X_negative_val))

a = list(np.ones(X_positive_val.shape[0]))
b = list(np.zeros(X_negative_val.shape[0]))
c = list(-np.ones(X_positive_test.shape[0] + X_negative_test.shape[0] + X_train.shape[0]))
a.extend(b)
#a.extend(c)

labels = np.array(a)


clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X, a) 

d = clf.predict(X_positive_test)
e = clf.predict(X_negative_test)
f = clf.predict(X_train)