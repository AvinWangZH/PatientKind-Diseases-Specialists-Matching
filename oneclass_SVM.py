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


with open('label.p', 'rb') as f:
    label = pickle.load(f)
    
    
with open('training_data.p', 'rb') as f:
    training = pickle.load(f)

X_train_ = np.zeros((label.size-2160, 2))
X_test_ = np.zeros((2160, 2))
X_outlier_ = np.zeros((2160, 2))
count = 0
count1 = 0

for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        if label[i][j] != 0:
            X_outlier_[count][0] = training[i][j][0]
            X_outlier_[count][1] = training[i][j][1]
            count += 1
        else:
            X_train_[count1][0] = training[i][j][0]
            X_train_[count1][1] = training[i][j][1]
            count1 += 1 
            
X_train = X_train_[1000:100000]
X_test = X_train_[:1000]
X_outlier = X_outlier_[:1000]
            
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
 
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outlier = clf.predict(X_outlier)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outlier = y_pred_outlier[y_pred_outlier == 1].size   
    

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

