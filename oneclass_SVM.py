"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================

An example using a one-class SVM for novelty detection.

:ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
algorithm that learns a decision function for novelty detection:
classifying new data as similar or different to the training set.
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import pickle
import json
import logging
      
with open('training_data_dict.json', 'r') as f:
    training_data_dict = json.load(f)
    
with open('positive_set.p', 'rb') as f:
    positive_set = pickle.load(f)
    
with open('negative_set.p', 'rb') as f:
    negative_set = pickle.load(f)

logging.basicConfig(level='INFO')
np.random.seed(1)

positive_random_row_num = np.random.choice(positive_set.shape[0], positive_set.shape[0], replace=False)
positive_val_row_num = positive_random_row_num[:int(positive_set.shape[0]/2)]
positive_test_row_num = positive_random_row_num[int(positive_set.shape[0]/2):]

negative_random_row_num = np.random.choice(negative_set.shape[0], negative_set.shape[0], replace=False)
negative_val_row_num = negative_random_row_num[:int(positive_set.shape[0]/2)]
negative_test_row_num = negative_random_row_num[int(positive_set.shape[0]/2):int(positive_set.shape[0])]
negative_training_row_num = negative_random_row_num[int(positive_set.shape[0]):]

X_positive_val = np.zeros((1, 11))
X_negative_val = np.zeros((1, 11))
X_positive_test = np.zeros((1, 11))
X_negative_test = np.zeros((1, 11))
X_train = np.zeros((1, 11))

count = 1
for row_num in positive_val_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_val = np.vstack((X_positive_val, positive_set[row_num, :]))
        count += 1
    if count == 1000:
        break
logging.info('X_positive_val has been finished')

count = 1
for row_num in negative_val_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_val = np.vstack((X_negative_val, negative_set[row_num, :]))
        count += 1
    if count == 1000:
        break        
logging.info('X_negative_val has been finished')
  
count = 1  
for row_num in positive_test_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_test = np.vstack((X_positive_test, positive_set[row_num, :]))
        count += 1
    if count == 1000:
        break        
logging.info('X_positive_test has been finished')

count = 1  
for row_num in negative_test_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_test = np.vstack((X_negative_test, negative_set[row_num, :]))
        count += 1
    if count == 1000:
        break        
logging.info('X_negative_test has been finished')

for row_num in negative_training_row_num[:50000]:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_train = np.vstack((X_train, negative_set[row_num, :]))
logging.info('X_train has been finished')
    
X_train = np.delete(X_train, 0, 0)
X_positive_val = np.delete(X_positive_val, 0, 0)
X_negative_val = np.delete(X_negative_val, 0, 0)
X_positive_test = np.delete(X_positive_test, 0, 0)
X_negative_test = np.delete(X_negative_test, 0, 0)


n_error_train_list = []
n_error_test_list = []
n_error_outlier_list = []
prec_list = []
rec_list = []
F1_list = []

value = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

for index, gamma_value in enumerate(value):
    n_error_train_list.append([])
    n_error_test_list.append([])
    n_error_outlier_list.append([]) 
    prec_list.append([])
    rec_list.append([])
    F1_list.append([])
    for nu_value in value[:7]:
        clf = svm.OneClassSVM(nu=nu_value, kernel="rbf", gamma=gamma_value)
        clf.fit(X_train)
        logging.info('Training with gamma {} and nu {} has been complished'.format(gamma_value, nu_value))
         
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_negative_val)
        y_pred_outlier = clf.predict(X_positive_val)
        n_train = X_train.shape[0]
        n_test = X_negative_val.shape[0]
        n_outlier = X_positive_val.shape[0]
        
        n_error_train_list[index].append(y_pred_train[y_pred_train == -1].size/n_train)
        n_error_test_list[index].append(y_pred_test[y_pred_test == -1].size/n_test)
        n_error_outlier_list[index].append(y_pred_outlier[y_pred_outlier == 1].size/n_outlier)
        
        print('False positive rate: %04.2f and False negative rate: %04.2f' %(y_pred_test[y_pred_test == -1].size/n_test, y_pred_outlier[y_pred_outlier == 1].size/n_outlier))
        
        tp = 1000 - y_pred_outlier[y_pred_outlier == 1].size
        fp = y_pred_test[y_pred_test == -1].size
        fn = y_pred_outlier[y_pred_outlier == 1].size
        
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        
        F1 = 2 * prec * rec / (prec + rec)
        
        prec_list[index].append(prec)
        rec_list[index].append(rec)
        F1_list[index].append(F1)
        
        print('Precision is: %f' % prec)
        print('Recall is: %f' % rec)
        print('F1 score is: %f' % F1)
        
        print('\n')

'''
xx, yy = np.meshgrid(np.linspace(-50, 50, 500), np.linspace(-50, 50, 500))
# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")


b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
c = plt.scatter(X_outlier[:, 0], X_outlier[:, 1], c='red')
plt.axis('tight')
plt.xlim((-50, 50))
plt.ylim((-50, 50))
plt.legend([ b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outlier))
plt.show()
'''



#-----------------------------------------------------Unused Code------------------------------------------------
'''
#1. get outlier data
X_outlier = np.zeros((1, 11))   #the first row is just for formatting
for omim_id in training_data_dict:
    for author in training_data_dict[omim_id]:
        if training_data_dict[omim_id][author][1] == 1 and not np.any(np.isnan(training_data_dict[omim_id][author][0])):
            X_outlier = np.vstack((X_outlier, training_data_dict[omim_id][author][0]))

count = 0
X_full = np.zeros((1, 11))
for omim_id in training_data_dict:
    for author in training_data_dict[omim_id]:
        if training_data_dict[omim_id][author][1] == 0 and not np.any(np.isnan(training_data_dict[omim_id][author][0])):
            X_full = np.vstack((X_full, training_data_dict[omim_id][author][0]))
            count+=1
            print(count)
            if count == 50000:
                break
    if count == 50000:
        break
            
X_train = np.concatenate((X_full[1000:49999, 0:1], X_full[1000:49999, 4:]), axis=1)
X_test = np.concatenate((X_full[1:1000, 0:1], X_full[1:1000, 4:]), axis=1)
X_outlier = np.concatenate((X_outlier[1:1000, 0:1], X_outlier[1:1000, 4:]), axis=1) #2160
'''
