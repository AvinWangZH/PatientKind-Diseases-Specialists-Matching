import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import json
import pickle
import logging
import seaborn as sns
from sklearn import cross_validation


with open('training_data_dict.json', 'r') as f:
    training_data_dict = json.load(f)
    
with open('positive_set_with_names_and_omimid.json', 'r') as f:
    positive_set_temp = json.load(f)
    
with open('negative_set_with_names_and_omimid.json', 'r') as f:
    negative_set_temp = json.load(f)
    
#with open('positive_set.p', 'rb') as f:
    #positive_set = pickle.load(f)
    
#with open('negative_set.p', 'rb') as f:
    #negative_set = pickle.load(f)

positive_set = []
positive_set_names_temp = []
for entry in positive_set_temp:
    positive_set.append(entry[:-2])
    positive_set_names_temp.append((entry[-2], entry[-1]))
positive_set = np.array(positive_set)

negative_set = []
negative_set_names_temp = []
for entry in negative_set_temp:
    negative_set.append(entry[:-2])
    negative_set_names_temp.append((entry[-2], entry[-1]))
negative_set = np.array(negative_set)


logging.basicConfig(level='INFO')
np.random.seed(1)

positive_random_row_num = np.random.choice(positive_set.shape[0], positive_set.shape[0], replace=False)
positive_val_row_num = positive_random_row_num[:int(positive_set.shape[0]*3/4)]
positive_test_row_num = positive_random_row_num[int(positive_set.shape[0]*3/4):]

negative_random_row_num = np.random.choice(negative_set.shape[0], negative_set.shape[0], replace=False)
negative_val_row_num = negative_random_row_num[:int(positive_set.shape[0]*3/4)]
negative_test_row_num = negative_random_row_num[int(positive_set.shape[0]*3/4):int(positive_set.shape[0])]
negative_training_row_num = negative_random_row_num[int(positive_set.shape[0]):]

num_features = 8
X_positive_val = [] 
X_negative_val = [] 
X_positive_test = [] 
X_negative_test = [] 
X_train = []

X_positive_val_names = []
X_negative_val_names = []
X_positive_test_names = []
X_negative_test_names = []
X_train_names = []

count = 1
for row_num in positive_val_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_val.append(list(np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 4:]), axis=0)))
        X_positive_val_names.append(positive_set_names_temp[row_num])
        count += 1
    if count == 1500:
        break
X_positive_val = np.array(X_positive_val)
logging.info('X_positive_val has been finished')

count = 1
for row_num in negative_val_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_val.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        X_negative_val_names.append(negative_set_names_temp[row_num])
        count += 1
    if count == 1500:
        break      
X_negative_val = np.array(X_negative_val)
logging.info('X_negative_val has been finished')

  
count = 1  
for row_num in positive_test_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_test.append(list(np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 4:]), axis=0)))
        X_positive_test_names.append(positive_set_names_temp[row_num])
        count += 1
    if count == 500:
        break       
X_positive_test = np.array(X_positive_test)
logging.info('X_positive_test has been finished')

count = 1  
for row_num in negative_test_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_test.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        X_negative_test_names.append(negative_set_names_temp[row_num])
        count += 1
    if count == 500:
        break     
X_negative_test = np.array(X_negative_test)
logging.info('X_negative_test has been finished')

for row_num in negative_training_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_train.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        X_train_names.append(negative_set_names_temp[row_num])
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


clf = SVC(probability=True)
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
            


pos_test = clf.predict_proba(X_positive_test)
neg_test = clf.predict_proba(X_negative_test)
left = clf.predict_proba(X_train)

#f, axarr = plt.subplots(2, 2)

#axarr[0, 0].boxplot(pos_1000[:, 1])
#axarr[0, 0].set_title('positive testing set')
#axarr[0, 1].boxplot(neg_1000[:, 1])
#axarr[0, 1].set_title('negative testing set')
#axarr[1, 0].boxplot(left[:, 1])
#axarr[1, 0].set_title('Rest unknown set')

fig = sns.distplot(pos_1000[:, 1]);
fig.set(title='Probability of Experts based on Positive Test Set', xlabel='Probability', ylabel='Percentage')

#sns.distplot(pos_1000[:, 1]);
#plt.show()


#--------------------------------Find Outlier By Hand and Test-----------------------

#The best performance happens on threshold = 0.4
cutoff = 0.5

index = 0
outlier_index_list = []
for score in neg_test[:, 1]:
    if score >= cutoff:
        outlier_index_list.append(index)
    index += 1

index = 0
index_list_p = []
for score in pos_test[:, 1]:
    if score >= cutoff:
        index_list_p.append(index)
    index += 1
    
fp = len(outlier_index_list)
fn = 500 - len(index_list_p)
tp = len(index_list_p)

prec = tp / (tp + fp)
rec = tp / (tp + fn)

F1 = 2 * prec * rec / (prec + rec)    

accuracy = 1 - (fp + fn)/1000

logging.info('Threshold: {}'.format(cutoff))
logging.info('False Positive Rate: {}'.format(fp/500))
logging.info('False Negative Rate: {}'.format(fn/500))
logging.info('Accuracy: {}'.format(accuracy))
logging.info('Precision: {}'.format(prec))
logging.info('Recall: {}'.format(rec))
logging.info('F1: {}'.format(F1))
print('\n')   
    
    
    
