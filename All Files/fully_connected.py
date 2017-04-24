import random
import json
import pickle
import logging

import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import *

import tensorflow as tf

from utils import run_single_method

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
    idx = np.random.choice(tr_x.shape[0], size, replace=True)
    return tr_x[idx, :], tr_y[idx, :]

def one_hot(y):
    # temp = []
    # for i in y:
    #     if i == 1:
    #         temp.append([1, 0])
    #     else:
    #         temp.append([0, 1])
    # return np.array(temp)
    return np.stack((1 - y, y), axis=1)

def run(X_train, y_train, X_test, seed):
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    Y_train = one_hot(y_train)

    # Parameters
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
    layer1 = tf.nn.relu(tf.matmul(x, W0) + b0)
    layer2 = tf.matmul(layer1, W1) + b1

    # Construct model
    pred = tf.nn.softmax(layer2) # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = get_batch(X_train, Y_train, batch_size)
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
        # print("Training Accuracy:", accuracy.eval({x: X_train, y: Y_train}))

        y_pred_prob = sess.run(pred, feed_dict = {x: X_test})[:, 1]
        return y_pred_prob

        # y_pos_full_pred = sess.run(pred, feed_dict = {x: positive_set})
        # y_neg_full_pred = sess.run(pred, feed_dict = {x: negative_set})
        # y_p = tf.argmin(pred, 1)
        # test_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:x_test, y:y_test})
        # y_true = np.argmin(y_test,1)


        # temp = y_true - y_pred
        # fn = list(temp).count(1)/(len(y_true) - sum(y_true))
        # #print(confusion_matrix(y_true, y_pred))
        # fpr, tpr, tresholds = roc_curve(y_true, y_pred)
        # logging.info('False Positive Rate: {}'.format(fpr[1]))
        # logging.info('False Negative Rate: {}'.format(fn))
        # print("Precision", precision_score(y_true, y_pred))
        # print("Recall", recall_score(y_true, y_pred))
        # print("f1_score", f1_score(y_true, y_pred))
        # #print("confusion_matrix")
        # print('\n')

        # fp_rate_list.append(fpr[1])
        # fn_rate_list.append(fn)
        # tp_rate_list.append(tpr[1])
        # accuracy_list.append(accuracy.eval({x: x_test, y: y_test}))
        # prec_list.append(precision_score(y_true, y_pred))
        # rec_list.append(recall_score(y_true, y_pred))
        # F1_list.append(f1_score(y_true, y_pred))

        # #Find ranks of each author
        # y_left_pred = sess.run(pred, feed_dict = {x: X_left})
        # percentage, author_num_mean, author_num_median = find_rank(y_left_pred, X_left_names, y_true, X_test_names, y_pred_prob)
        # #rank_percentage_list.append(percentage)
        # #author_num_mean_list.append(author_num_mean)
        # #author_num_median_list.append(author_num_median)

        # print(percentage, author_num_mean, author_num_median)
        # print('\n')
        # break


if __name__ == '__main__':
    run_single_method(run)

    # #Plot the histogram
    # pos_test = []
    # neg_test = []

    # #Plot the graph
    # for index, label in enumerate(list(y_true)):
    #     if label == 1:
    #         pos_test.append(y_pred_prob[index, 0])
    #     else:
    #         neg_test.append(y_pred_prob[index, 0])
    # f, (ax1, ax2) = plt.subplots(1,2)
    # fig1 = sns.distplot(pos_test, bins = 20, norm_hist = False, kde=False, ax=ax1);
    # fig1.set(title='Distribution of posterior probabilities (positive test set)', xlabel='Probability', ylabel='Count')
    # fig2 = sns.distplot(neg_test, bins = 20, norm_hist = False, kde=False, ax=ax2);
    # fig2.set(title='Distribution of posterior probabilities (negative test set)', xlabel='Probability', ylabel='Count')
    # plt.show()


    # temp_pos_test = pos_test
    # temp_pos_test = sorted(list(temp_pos_test))
    # temp_pos_test.reverse()
    # q1_pos_test = temp_pos_test[0]
    # q2_pos_test = temp_pos_test[int(len(temp_pos_test)/4)]
    # q3_pos_test = temp_pos_test[int(len(temp_pos_test)/2)]
    # q4_pos_test = temp_pos_test[int(len(temp_pos_test)/4*3)]
    # q5_pos_test = temp_pos_test[-1]

    # temp_neg_test = neg_test
    # temp_neg_test = sorted(list(temp_neg_test))
    # temp_neg_test.reverse()
    # q1_neg_test = temp_neg_test[0]
    # q2_neg_test = temp_neg_test[int(len(temp_pos_test)/4)]
    # q3_neg_test = temp_neg_test[int(len(temp_pos_test)/2)]
    # q4_neg_test = temp_neg_test[int(len(temp_pos_test)/4*3)]
    # q5_neg_test = temp_neg_test[-1]

    # #-----------------------------------Graphing----------------------------------
    # f, axarr = plt.subplots(1, 2)

    # axarr[0].boxplot(y_pos_full_pred[:, 0])
    # axarr[0].set_title('positive set (full)')
    # axarr[1].boxplot(y_neg_full_pred[:, 0])
    # axarr[1].set_title('negative set/unknown (full)')

    # plt.show()


    # temp_pos = y_pos_full_pred[:, 0]
    # temp_pos = sorted(list(temp_pos))
    # temp_pos.reverse()
    # q1_pos = temp_pos[0]
    # q2_pos = temp_pos[int(len(temp_pos)/4)]
    # q3_pos = temp_pos[int(len(temp_pos)/2)]
    # q4_pos = temp_pos[int(len(temp_pos)*3/4)]
    # q5_pos = temp_pos[-1]

    # temp_neg = y_neg_full_pred[:, 0]
    # temp_neg = sorted(list(temp_neg))
    # temp_neg.reverse()
    # q1_neg = temp_neg[0]
    # q2_neg = temp_neg[int(len(temp_neg)/4)]
    # q3_neg = temp_neg[int(len(temp_neg)/2)]
    # q4_neg = temp_neg[int(len(temp_neg)*3/4)]
    # q5_neg = temp_neg[-1]


    # rank_dict = get_rank_list(y_left_pred, X_left_names, y_true, X_test_names, y_pred_prob)
