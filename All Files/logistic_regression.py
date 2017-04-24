import logging

from sklearn.linear_model import LogisticRegression

from utils import run_single_method

def run(X_train, y_train, X_test, seed):
    # Train
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Test
    y_score = clf.predict_proba(X_test)[:, 1]
    return y_score

if __name__ == '__main__':
    run_single_method(run)
