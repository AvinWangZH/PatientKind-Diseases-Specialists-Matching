print(__doc__)

import json
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import label_propagation


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

for row_num in negative_training_row_num[:10000]:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_train.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        #X_train.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
X_train = np.array(X_train)
logging.info('X_train has been finished')
    
#X_train = np.delete(X_train, 0, 0)
#X_positive_val = np.delete(X_positive_val, 0, 0)
#X_negative_val = np.delete(X_negative_val, 0, 0)
#X_positive_test = np.delete(X_positive_test, 0, 0)
#X_negative_test = np.delete(X_negative_test, 0, 0)


# generate ring with inner box
n_samples = X_positive_val.shape[0] + X_negative_val.shape[0] + X_positive_test.shape[0] + X_negative_test.shape[0] + X_train.shape[0]
X = np.vstack((X_positive_val, X_negative_val, X_positive_test, X_negative_test, X_train))

a = list(np.ones(X_positive_val.shape[0]))
b = list(np.zeros(X_negative_val.shape[0]))
c = list(-np.ones(X_positive_test.shape[0] + X_negative_test.shape[0] + X_train.shape[0]))
a.extend(b)
a.extend(c)

labels = np.array(a)


###############################################################################
# Learn with LabelSpreading

gamma_value = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

tp_list = []
fp_list = []
fn_list = []
prec_list = []
rec_list = []
accuracy_list = []
F1_list = []


for g in gamma_value:
    label_spread = label_propagation.LabelSpreading(kernel='rbf', gamma=g)
    label_spread.fit(X, labels)
    
    output_labels = label_spread.transduction_
    
    
    logging.info('For gamma: {}'.format(g))
    logging.info('False positive rate: {}'.format(output_labels[3000:4000].sum()/1000))
    logging.info('False negative rate: {}'.format(1 - output_labels[2000:3000].sum()/1000))
    
    tp = output_labels[2000:3000].sum()  
    fp = output_labels[3000:4000].sum()
    fn = 1000 - output_labels[2000:3000].sum()  
    
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    
    F1 = 2 * prec * rec / (prec + rec)    
    
    accuracy = 1 - (fp + fn)/2000
    
    tp_list.append(tp)
    fp_list.append(fp)
    fn_list.append(fn)
    prec_list.append(prec)
    rec_list.append(rec)
    accuracy_list.append(accuracy)
    F1_list.append(F1)

    logging.info('Accuracy: {}'.format(accuracy))
    logging.info('Precision: {}'.format(prec))
    logging.info('Recall: {}'.format(rec))
    logging.info('F1: {}'.format(F1))
    print('\n')
    
    #with open('30000_feature_1_3-11_s1_tp.json', 'w') as f:
        #json.dump(tp_list, f)
    
    #with open('30000_feature_1_3-11_s1_fp.json', 'w') as f:
        #json.dump(fp_list, f)    
    
    #with open('30000_feature_1_3-11_s1_fn.json', 'w') as f:
        #json.dump(fn_list, f)  
        
    #with open('30000_feature_1_3-11_s1_prec.json', 'w') as f:
        #json.dump(prec_list, f)    
    
    #with open('30000_feature_1_3-11_s1_rec.json', 'w') as f:
        #json.dump(rec_list, f)    
        
    #with open('30000_feature_1_3-11_s1_acc.json', 'w') as f:
        #json.dump(accuracy_list, f)   
        
    #with open('30000_feature_1_3-11_s1_F1.json', 'w') as f:
        #json.dump(F1_list, f)    
    
    

