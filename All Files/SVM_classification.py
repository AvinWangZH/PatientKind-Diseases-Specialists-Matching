import logging

from sklearn.svm import SVC

from utils import run_single_method

def run(X_train, y_train, X_test, seed):
    # Train
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)

    # Test
    y_score = clf.predict_proba(X_test)[:, 1]
    return y_score

if __name__ == '__main__':
    run_single_method(run)
