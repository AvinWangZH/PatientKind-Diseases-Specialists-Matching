import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import pickle
import logging
import seaborn as sns
from sklearn.cross_validation import KFold
from sklearn import grid_search
from sklearn.feature_selection import VarianceThreshold
import statistics

np.random.seed(1)

def find_rank(clf, X_left, X_left_names, y_test, X_test_names, X_pred_proba):
    X_left_pred_proba = clf.predict_proba(X_left)
    
    count = 0
    count_pos = 0
    number_of_author_list = []
    for index, label in enumerate(y_test):
        if label == 1:
            target_omim_id = X_test_names[index][1]
            target_prob = X_pred_proba[index][1]
            prob_list = [X_pred_proba[index][1]]
            for left_ind, name in enumerate(X_left_names):
                if X_left_names[left_ind][1] == target_omim_id:
                    prob_list.append(X_left_pred_proba[left_ind][1])
            prob_list.sort()
            prob_list.reverse()
            number_of_author_list.append(len(prob_list))
            #if (prob_list.index(target_prob)+1)/len(prob_list) < 0.3:
            #it will show the people who are at top 20
            if prob_list.index(target_prob) < 10:
                count += 1
           
    percentage = count/y_test.sum()
    author_num_mean = sum(number_of_author_list)/y_test.sum()
    author_num_median = statistics.median(number_of_author_list)
    
    return percentage, author_num_mean, author_num_median

with open('positive_set.json', 'r') as f:
    positive_set_temp = json.load(f)
    
with open('negative_set.json', 'r') as f:
    negative_set_temp = json.load(f)
    
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

positive_random_row_num = np.random.choice(positive_set.shape[0], positive_set.shape[0], replace=False)
positive_val_row_num = positive_random_row_num[:int(positive_set.shape[0]*1/2)]
positive_test_row_num = positive_random_row_num[int(positive_set.shape[0]*1/2):]

negative_random_row_num = np.random.choice(negative_set.shape[0], negative_set.shape[0], replace=False)
negative_val_row_num = negative_random_row_num[:int(positive_set.shape[0]*1/2)]
negative_test_row_num = negative_random_row_num[int(positive_set.shape[0]*1/2):int(positive_set.shape[0])]
negative_training_row_num = negative_random_row_num[int(positive_set.shape[0]):]

X_positive_val = [] 
X_negative_val = [] 
X_positive_test = [] 
X_negative_test = [] 
X_left = []

X_positive_val_names = []
X_negative_val_names = []
X_positive_test_names = []
X_negative_test_names = []
X_left_names = []

count = 1
for row_num in positive_val_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_val.append(list(np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 1:]), axis=0)))
        X_positive_val_names.append(positive_set_names_temp[row_num])
    if count == 1080:
        break
    count += 1
X_positive_val = np.array(X_positive_val)
logging.info('X_positive_val has been finished')

count = 1
for row_num in negative_val_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_val.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 1:]), axis=0)))
        X_negative_val_names.append(negative_set_names_temp[row_num])
    if count == 1080:
        break
    count += 1
X_negative_val = np.array(X_negative_val)
logging.info('X_negative_val has been finished')

  
count = 1  
for row_num in positive_test_row_num:
    if not np.any(np.isnan(positive_set[row_num, :])):
        X_positive_test.append(list(np.concatenate((positive_set[row_num, 0:1], positive_set[row_num, 1:]), axis=0)))
        X_positive_test_names.append(positive_set_names_temp[row_num])
    if count == 1080:
        break  
    count += 1
X_positive_test = np.array(X_positive_test)
logging.info('X_positive_test has been finished')

count = 1  
for row_num in negative_test_row_num:
    if not np.any(np.isnan(negative_set[row_num, :])):
        X_negative_test.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 1:]), axis=0)))
        X_negative_test_names.append(negative_set_names_temp[row_num])
    if count == 1080:
        break 
    count += 1
X_negative_test = np.array(X_negative_test)
logging.info('X_negative_test has been finished')

for row_num in negative_training_row_num:
    X_left.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 1:]), axis=0)))
    X_left_names.append(negative_set_names_temp[row_num])
X_left = np.array(X_left)
logging.info('X_left has been finished')


# generate training set
n_samples = X_positive_val.shape[0] + X_negative_val.shape[0] + X_positive_test.shape[0] + X_negative_test.shape[0]
names = np.vstack((X_positive_val_names, X_negative_val_names, X_positive_test_names, X_negative_test_names))
X = np.vstack((X_positive_val, X_negative_val, X_positive_test, X_negative_test))

# feature selection
selector = VarianceThreshold()
X = selector.fit_transform(X)

# generate labels
a = list(np.ones(X_positive_val.shape[0]))
b = list(np.zeros(X_negative_val.shape[0]))
c = list(np.ones(X_positive_test.shape[0]))
d = list(np.zeros(X_negative_test.shape[0]))
a.extend(b)
a.extend(c)
a.extend(d)
labels = np.array(a)


fp_rate_mean_list = []
fn_rate_mean_list = []
accuracy_mean_list = []
prec_mean_list = []
rec_mean_list = []
F1_mean_list = []

rank_percentage_mean_list = []
author_num_mean_mean_list = []
author_num_median_mean_list = []

#---------------------------------------Cross Validation----------------------------------------
#param_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
#Best C: 0.3 or 1

random_state_num = np.arange(1)
for num in random_state_num:
    count = 0
    kf = KFold(n_samples, n_folds=10,shuffle=True, random_state=num)
    
    cutoff = 0.5
    fp_rate_list = []
    fn_rate_list = []
    accuracy_list = []
    prec_list = []
    rec_list = []
    F1_list = []
    rank_percentage_list = []
    author_num_mean_list = []
    author_num_median_list = []    
    count = 0
    for train_index, test_index in kf:
        count += 1
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        X_train_names, X_test_names = names[train_index], names[test_index]
        
        clf = LogisticRegression()
        clf.fit(X_train, y_train) 
                
        X_pred_proba = clf.predict_proba(X_test)
        X_pred = []
        
        for score in X_pred_proba[:, 1]:
            if score >= cutoff:
                X_pred.append(1)
            else:
                X_pred.append(0)
        
        temp = y_test - X_pred
        
        fp = list(temp).count(-1)
        fn = list(temp).count(1)
        tp = sum(y_test) - fn
        
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        
        F1 = 2 * prec * rec / (prec + rec)    
        
        accuracy = 1 - (fp + fn)/(len(y_test))
        
        logging.info('Random Num: {}'.format(num))
        logging.info('Iteration: {}'.format(count))
        logging.info('False Positive Rate: {}'.format(fp/(sum(y_test))))
        logging.info('False Negative Rate: {}'.format(fn/(len(y_test) - sum(y_test))))
        logging.info('Accuracy: {}'.format(accuracy))
        logging.info('Precision: {}'.format(prec))
        logging.info('Recall: {}'.format(rec))
        logging.info('F1: {}'.format(F1))
        
        
        fp_rate_list.append(fp/(sum(y_test)))
        fn_rate_list.append(fn/(len(y_test) - sum(y_test)))
        accuracy_list.append(accuracy)
        prec_list.append(prec)
        rec_list.append(rec)
        F1_list.append(F1)
        
        #Find ranks of each author
        percentage, author_num_mean, author_num_median = find_rank(clf, X_left, X_left_names, y_test, X_test_names, X_pred_proba)
        rank_percentage_list.append(percentage)
        author_num_mean_list.append(author_num_mean)
        author_num_median_list.append(author_num_median) 
        
        print(percentage, author_num_mean, author_num_median)
        print('\n')
        
    fp_rate_list = np.array(fp_rate_list)
    fn_rate_list = np.array(fn_rate_list)
    accuracy_list = np.array(accuracy_list)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    F1_list = np.array(F1_list)
    rank_percentage_list = np.array(rank_percentage_list)
    author_num_mean_list = np.array(author_num_mean_list)
    author_num_median_list = np.array(author_num_median_list)     
    
    print('Mean Values:')
    logging.info('False Positive Rate: {}'.format(fp_rate_list.mean()))
    logging.info('False Negative Rate: {}'.format(fn_rate_list.mean()))
    logging.info('Accuracy: {}'.format(accuracy_list.mean()))
    logging.info('Precision: {}'.format(prec_list.mean()))
    logging.info('Recall: {}'.format(rec_list.mean()))
    logging.info('F1: {}'.format(F1_list.mean()))
    print('\n')
    
    fp_rate_mean_list.append(fp_rate_list.mean())
    fn_rate_mean_list.append(fn_rate_list.mean())
    accuracy_mean_list.append(accuracy_list.mean())
    prec_mean_list.append(prec_list.mean())
    rec_mean_list.append(rec_list.mean())
    F1_mean_list.append(F1_list.mean()) 
    rank_percentage_mean_list.append(rank_percentage_list.mean())
    author_num_mean_mean_list.append(author_num_mean_list.mean())
    author_num_median_mean_list.append(author_num_median_list.mean())    

#logging.info('C value: {}'.format(c))
print('Overall Mean Values:')
logging.info('False Positive Rate: {}'.format(np.array(fp_rate_mean_list).mean()))
logging.info('False Negative Rate: {}'.format(np.array(fn_rate_mean_list).mean()))
logging.info('Accuracy: {}'.format(np.array(accuracy_mean_list).mean()))
logging.info('Precision: {}'.format(np.array(prec_mean_list).mean()))
logging.info('Recall: {}'.format(np.array(rec_mean_list).mean()))
logging.info('F1: {}'.format(np.array(F1_mean_list).mean()))
logging.info('Rank%: {}'.format(np.array(rank_percentage_mean_list).mean()))
logging.info('#Auhtor Mean: {}'.format(np.array(author_num_mean_mean_list).mean()))
logging.info('#Author Median: {}'.format(np.array(author_num_median_mean_list).mean()))
print('\n')
 
            
#-----------------------------------Graphing----------------------------------
#f, axarr = plt.subplots(2, 2)

#axarr[0, 0].boxplot(pos_1000[:, 1])
#axarr[0, 0].set_title('positive testing set')
#axarr[0, 1].boxplot(neg_1000[:, 1])
#axarr[0, 1].set_title('negative testing set')
#axarr[1, 0].boxplot(left[:, 1])
#axarr[1, 0].set_title('Rest unknown set')

#fig = sns.distplot(pos_1000[:, 1]);
#fig.set(title='Probability of Experts based on Positive Test Set', xlabel='Probability', ylabel='Percentage')

#sns.distplot(pos_1000[:, 1]);
#plt.show()


#--------------------------------Find Outlier By Hand and Test-----------------------

#The best performance happens on threshold = 0.4
#cutoff = 0.5

#index = 0
#outlier_index_list = []
#for score in neg_test[:, 1]:
    #if score >= cutoff:
        #outlier_index_list.append(index)
    #index += 1

#index = 0
#index_list_p = []
#for score in pos_test[:, 1]:
    #if score >= cutoff:
        #index_list_p.append(index)
    #index += 1
    
#fp = len(outlier_index_list)
#fn = 200 - len(index_list_p)
#tp = len(index_list_p)

#prec = tp / (tp + fp)
#rec = tp / (tp + fn)

#F1 = 2 * prec * rec / (prec + rec)    

#accuracy = 1 - (fp + fn)/400

#logging.info('Threshold: {}'.format(cutoff))
#logging.info('False Positive Rate: {}'.format(fp/200))
#logging.info('False Negative Rate: {}'.format(fn/200))
#logging.info('Accuracy: {}'.format(accuracy))
#logging.info('Precision: {}'.format(prec))
#logging.info('Recall: {}'.format(rec))
#logging.info('F1: {}'.format(F1))
#print('\n')   
    
    
    
