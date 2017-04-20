import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
import numpy as np
import json
import pickle
import logging
import seaborn as sns
from sklearn.cross_validation import KFold
import tensorflow as tf
import random
from sklearn.metrics import *
import statistics

def find_rank(X_left_pred_proba, X_left_names, y_test, X_test_names, X_pred_proba):
    count = 0
    count_pos = 0
    number_of_author_list = []
    for index, label in enumerate(y_test):
        if label == 1:
            target_omim_id = X_test_names[index][1]
            target_prob = X_pred_proba[index][0]
            prob_list = [X_pred_proba[index][0]]
            for left_ind, name in enumerate(X_left_names):
                if X_left_names[left_ind][1] == target_omim_id:
                    prob_list.append(X_left_pred_proba[left_ind][0])
            prob_list.sort()
            prob_list.reverse()
            number_of_author_list.append(len(prob_list))
            #if (prob_list.index(target_prob)+1)/len(prob_list) < 0.3:
            #it will show the people who are at top 10
            if prob_list.index(target_prob) < 5:
                count += 1
           
    percentage = count/y_test.sum()
    author_num_mean = sum(number_of_author_list)/y_test.sum()
    author_num_median = statistics.median(number_of_author_list)
    
    return percentage, author_num_mean, author_num_median

def get_rank_list(X_left_pred_proba, X_left_names, y_test, X_test_names, X_pred_proba):
    all_pred = []
    all_names = []
    omim_author_rank_dict = {}
    
    for i in X_left_pred_proba:
        all_pred.append(i)
    for i in X_pred_proba:
        all_pred.append(i)
    for i in X_left_names:
        all_names.append(i)
    for i in X_test_names:
        all_names.append(i)
    
    for s in all_names:
        omim_author_rank_dict[s[1]] = [[], []]
    
    for omim in omim_author_rank_dict:
        for i in range(len(all_names)):
            if all_names[i][1] == omim:
                omim_author_rank_dict[omim][0].append(all_names[i][0])
                omim_author_rank_dict[omim][1].append(all_pred[i][0])
    for omim in omim_author_rank_dict:
        list1, list2 = zip(*sorted(zip(omim_author_rank_dict[omim][1], omim_author_rank_dict[omim][0])))
        list1, list2 = list(list1), list(list2)
        list1.reverse()
        list2.reverse()
        omim_author_rank_dict[omim] = [list2, list1]
    return omim_author_rank_dict

def get_batch(tr_x, tr_y, size):
    rand_ind = random.sample(range(1, tr_x.shape[0]), size)
    temp_x = []
    temp_y = []
    for i in rand_ind:
        temp_x.append(tr_x[i])
        temp_y.append(tr_y[i])
    return np.array(temp_x), np.array(temp_y)

def one_hot(y):
    temp = []
    for i in y:
        if i == 1:
            temp.append([1, 0])
        else:
            temp.append([0, 1])
    return np.array(temp)
    
with open('positive_set.json', 'r') as f:
    positive_set_temp = json.load(f)
    
with open('negative_set.json', 'r') as f:
    negative_set_temp = json.load(f)
    

np.random.seed(2)
tf.set_random_seed(2)
random.seed(2)

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
np.random.seed(2)

positive_random_row_num = np.random.choice(positive_set.shape[0], positive_set.shape[0], replace=False)
positive_val_row_num = positive_random_row_num[:int(positive_set.shape[0]*1/2)]
positive_test_row_num = positive_random_row_num[int(positive_set.shape[0]*1/2):]

negative_random_row_num = np.random.choice(negative_set.shape[0], negative_set.shape[0], replace=False)
negative_val_row_num = negative_random_row_num[:int(positive_set.shape[0]*1/2)]
negative_test_row_num = negative_random_row_num[int(positive_set.shape[0]*1/2):int(positive_set.shape[0])]
negative_training_row_num = negative_random_row_num[int(positive_set.shape[0]):]

num_features = 18
X_positive_val = [] 
X_negative_val = [] 
X_positive_test = [] 
X_negative_test = [] 
X_train = []
X_left = []

X_positive_val_names = []
X_negative_val_names = []
X_positive_test_names = []
X_negative_test_names = []
X_train_names = []
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

#for row_num in negative_training_row_num:
    #if not np.any(np.isnan(negative_set[row_num, :])):
        #X_train.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 4:]), axis=0)))
        #X_train_names.append(negative_set_names_temp[row_num])
#X_train = np.array(X_train)
#logging.info('X_train has been finished')

for row_num in negative_training_row_num:
    X_left.append(list(np.concatenate((negative_set[row_num, 0:1], negative_set[row_num, 1:]), axis=0)))
    X_left_names.append(negative_set_names_temp[row_num])
X_left = np.array(X_left)
logging.info('X_left has been finished')

# generate training set
n_samples = X_positive_val.shape[0] + X_negative_val.shape[0] + X_positive_test.shape[0] + X_negative_test.shape[0]
names = np.vstack((X_positive_val_names, X_negative_val_names, X_positive_test_names, X_negative_test_names))
X = np.vstack((X_positive_val, X_negative_val, X_positive_test, X_negative_test))

# generate labels
a = list(np.ones(X_positive_val.shape[0]))
b = list(np.zeros(X_negative_val.shape[0]))
c = list(np.ones(X_positive_test.shape[0]))
d = list(np.zeros(X_negative_test.shape[0]))
a.extend(b)
a.extend(c)
a.extend(d)
labels = np.array(a)

Y = one_hot(labels)

#t_x = np.vstack((X[0:100], X[-100:-1]))
#t_y = np.vstack((Y[0:100], Y[-100:-1]))

#tr_x = X[100:4900]
#tr_y = Y[100:4900]
# Parameters
   

fp_rate_mean_list = []
fn_rate_mean_list = []
tp_rate_mean_list = []
accuracy_mean_list = []
prec_mean_list = []
rec_mean_list = []
F1_mean_list = []

#random_state_num = np.arange(1)
#for num in random_state_num:
#count = 0
learning_rate = 0.0004
training_epochs = 500
batch_size = 2000
display_step = 2000
nhid = 300

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 18]) 
y = tf.placeholder(tf.float32, [None, 2])

# Set model weights

W0 = tf.Variable(tf.random_normal([18, nhid], stddev=0.0001))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.0001))

W1 = tf.Variable(tf.random_normal([nhid, 2], stddev=0.0001))
b1 = tf.Variable(tf.random_normal([2], stddev=0.0001))

#Define the layers
layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


# Construct model
pred = tf.nn.softmax(layer2) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer() 

#Define 10 folds
kf = KFold(n_samples, n_folds=10,shuffle=True, random_state=1)

fp_rate_list = []
fn_rate_list = []
tp_rate_list = []
accuracy_list = []
prec_list = []
rec_list = []
F1_list = []
count = 0
for train_index, test_index in kf:
    count += 1
    logging.info('Iteration: {}'.format(count))
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    X_train_names, X_test_names = names[train_index], names[test_index]
    

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
    
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X.shape[0]/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = get_batch(x_train, y_train, batch_size)
                # Fit training using batch data
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    
        print("Optimization Finished!")
    
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Training Accuracy:", accuracy.eval({x: x_train, y: y_train}))  
        print("Test Accuracy:", accuracy.eval({x: x_test, y: y_test})) 
        
        y_pred_prob = sess.run(pred, feed_dict = {x: x_test})
        y_pos_full_pred = sess.run(pred, feed_dict = {x: positive_set})
        y_neg_full_pred = sess.run(pred, feed_dict = {x: negative_set})        
        y_p = tf.argmin(pred, 1)
        test_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:x_test, y:y_test})  
        y_true = np.argmin(y_test,1)
        temp = y_true - y_pred
        fn = list(temp).count(1)/(len(y_true) - sum(y_true))
        #print(confusion_matrix(y_true, y_pred))
        fpr, tpr, tresholds = roc_curve(y_true, y_pred) 
        logging.info('False Positive Rate: {}'.format(fpr[1]))    
        logging.info('False Negative Rate: {}'.format(fn))  
        print("Precision", precision_score(y_true, y_pred))
        print("Recall", recall_score(y_true, y_pred))
        print("f1_score", f1_score(y_true, y_pred))
        #print("confusion_matrix")
        print('\n')
        
        fp_rate_list.append(fpr[1])
        fn_rate_list.append(fn)
        tp_rate_list.append(tpr[1])
        accuracy_list.append(accuracy.eval({x: x_test, y: y_test}))
        prec_list.append(precision_score(y_true, y_pred))
        rec_list.append(recall_score(y_true, y_pred))
        F1_list.append(f1_score(y_true, y_pred))
        
        #Find ranks of each author
        y_left_pred = sess.run(pred, feed_dict = {x: X_left})
        percentage, author_num_mean, author_num_median = find_rank(y_left_pred, X_left_names, y_true, X_test_names, y_pred_prob)
        #rank_percentage_list.append(percentage)
        #author_num_mean_list.append(author_num_mean)
        #author_num_median_list.append(author_num_median) 
        
        print(percentage, author_num_mean, author_num_median)
        print('\n')        
        break
    
fp_rate_list = np.array(fp_rate_list)
fn_rate_list = np.array(fn_rate_list)
tp_rate_list = np.array(tp_rate_list)
accuracy_list = np.array(accuracy_list)
prec_list = np.array(prec_list)
rec_list = np.array(rec_list)
F1_list = np.array(F1_list)

print('Mean Values:')
logging.info('False Positive Rate: {}'.format(fp_rate_list.mean()))
logging.info('False Negative Rate: {}'.format(fn_rate_list.mean()))
logging.info('True Positive Rate: {}'.format(tp_rate_list.mean()))
logging.info('Accuracy: {}'.format(accuracy_list.mean()))
logging.info('Precision: {}'.format(prec_list.mean()))
logging.info('Recall: {}'.format(rec_list.mean()))
logging.info('F1: {}'.format(F1_list.mean()))
print('\n')        


#Plot the histogram      
pos_test = []
neg_test = []

#Plot the graph
for index, label in enumerate(list(y_true)):
    if label == 1:
        pos_test.append(y_pred_prob[index, 0])
    else:
        neg_test.append(y_pred_prob[index, 0])
f, (ax1, ax2) = plt.subplots(1,2)
fig1 = sns.distplot(pos_test, bins = 20, norm_hist = False, kde=False, ax=ax1);
fig1.set(title='Distribution of posterior probabilities (positive test set)', xlabel='Probability', ylabel='Count')
fig2 = sns.distplot(neg_test, bins = 20, norm_hist = False, kde=False, ax=ax2);
fig2.set(title='Distribution of posterior probabilities (negative test set)', xlabel='Probability', ylabel='Count')
plt.show() 


temp_pos_test = pos_test
temp_pos_test = sorted(list(temp_pos_test))
temp_pos_test.reverse()
q1_pos_test = temp_pos_test[0]
q2_pos_test = temp_pos_test[int(len(temp_pos_test)/4)]
q3_pos_test = temp_pos_test[int(len(temp_pos_test)/2)]
q4_pos_test = temp_pos_test[int(len(temp_pos_test)/4*3)]
q5_pos_test = temp_pos_test[-1]

temp_neg_test = neg_test
temp_neg_test = sorted(list(temp_neg_test))
temp_neg_test.reverse()
q1_neg_test = temp_neg_test[0]
q2_neg_test = temp_neg_test[int(len(temp_pos_test)/4)]
q3_neg_test = temp_neg_test[int(len(temp_pos_test)/2)]
q4_neg_test = temp_neg_test[int(len(temp_pos_test)/4*3)]
q5_neg_test = temp_neg_test[-1]

#-----------------------------------Graphing----------------------------------
f, axarr = plt.subplots(1, 2)

axarr[0].boxplot(y_pos_full_pred[:, 0])
axarr[0].set_title('positive set (full)')
axarr[1].boxplot(y_neg_full_pred[:, 0])
axarr[1].set_title('negative set/unknown (full)')

plt.show()


temp_pos = y_pos_full_pred[:, 0]
temp_pos = sorted(list(temp_pos))
temp_pos.reverse()
q1_pos = temp_pos[0]
q2_pos = temp_pos[539]
q3_pos = temp_pos[539+540]
q4_pos = temp_pos[539+540+540]
q5_pos = temp_pos[-1]

temp_neg = y_neg_full_pred[:, 0]
temp_neg = sorted(list(temp_neg))
temp_neg.reverse()
q1_neg = temp_neg[0]
q2_neg = temp_neg[int(206950/4)]
q3_neg = temp_neg[int(206950/2)]
q4_neg = temp_neg[int(206950/4*3)]
q5_neg = temp_neg[-1]
 
 
rank_dict = get_rank_list(y_left_pred, X_left_names, y_true, X_test_names, y_pred_prob) 