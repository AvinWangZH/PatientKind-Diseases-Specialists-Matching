import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve as _roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def pr_curve(y_test, y_score):
    precision, recall, thresholds = precision_recall_curve(y_test, y_score, pos_label=1)
    average_precision = average_precision_score(y_test, y_score)
    lw = 2
    plt.clf()
    plt.plot(recall, precision, lw=lw, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('AUC={0:0.2f}'.format(average_precision))
    plt.legend(loc="lower left")
    plt.show()
    return average_precision

def roc_curve(y_test, y_score):
    fpr, tpr, thresholds = _roc_curve(y_test, y_score, pos_label=1)
    auc_score = auc(fpr, tpr)
    lw = 2
    plt.clf()
    plt.plot(fpr, tpr, lw=lw, color='navy',
             label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('AUC={0:0.2f}'.format(auc_score))
    plt.legend(loc="lower left")
    plt.show()
    return auc_score


# #Plot the graph
# for index, label in enumerate(y_test):
#     if label == 1:
#         pos_test.append(X_pred_proba[index, 1])
#     else:
#         neg_test.append(X_pred_proba[index, 1])
# f, (ax1, ax2) = plt.subplots(1,2)
# fig1 = sns.distplot(pos_test, bins = 20, norm_hist = False, kde=False, ax=ax1);
# fig1.set(title='Distribution of posterior probabilities (positive test set)', xlabel='Probability', ylabel='Count')
# fig2 = sns.distplot(neg_test, bins = 20, norm_hist = False, kde=False, ax=ax2);
# fig2.set(title='Distribution of posterior probabilities (negative test set)', xlabel='Probability', ylabel='Count')
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



