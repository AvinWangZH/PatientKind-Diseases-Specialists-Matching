import json
import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from baseline import run as run_baseline
from naive_bayes import run as run_nb
from random_forest import run as run_rf
from SVM_classification import run as run_svm
from fully_connected import run as run_nnet

from utils import parse_dataset
from utils import get_training_set
from utils import run_with_cv


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    with open('positive_set.json', 'r') as f:
        positive_set, positive_meta = parse_dataset(json.load(f))

    with open('negative_set.json', 'r') as f:
        negative_set, negative_meta = parse_dataset(json.load(f))

    seed = 1
    methods = [
        ('baseline', run_baseline, 'navy'),
        ('naive bayes', run_nb, 'turquoise'),
        ('random forest', run_rf, 'darkorange'),
        ('svm', run_svm, 'cornflowerblue'),
        ('neural net', run_nnet, 'teal'),
    ]

    X, y, meta = get_training_set(positive_set, positive_meta, negative_set, negative_meta, seed)
    groups = meta[:, 1]  # omim id

    plt.clf()
    for i, (method, run_method, color) in enumerate(methods):
        logging.info('Running method: {}'.format(method))
        y_test, y_score = run_with_cv(run_method, X, y, groups, seed)

        fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=color,
             label='ROC curve of {} (area = {:0.2f}'.format(method, auc_score))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Performances')
    plt.legend(loc="lower right")
    plt.show()

        # precision, recall, thresholds = precision_recall_curve(y_test, y_score, pos_label=1)
        # average_precision = average_precision_score(y_test, y_score)
        # plt.plot(recall, precision, color=color, lw=2,
        #      label='PR curve of {0} (area = {1:0.2f})'
        #            ''.format(method, average_precision))

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Performances')
    # plt.legend(loc="lower right")
    # plt.show()
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



