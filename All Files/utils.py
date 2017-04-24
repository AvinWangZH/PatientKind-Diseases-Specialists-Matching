import logging
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold

from display_results import pr_curve
from display_results import roc_curve

def parse_dataset(dataset):
    data = []
    metadata = []
    for row in dataset:
        # Drop the author name (-2) and omim id (-1)
        data.append(row[:-2])
        metadata.append(row[-2:])

    X = np.matrix(data)
    assert not np.any(np.isnan(X))
    assert X.shape[1] == 18 # 18 feature columns
    meta = np.matrix(metadata)
    return X, meta


def get_training_set(pos, pos_meta, neg, neg_meta, seed):
    np.random.seed(seed)

    N = len(pos)
    pos_idx = np.random.choice(len(pos), N, replace=False)
    neg_idx = np.random.choice(len(neg), N, replace=False)

    X = np.concatenate((pos[pos_idx, :], neg[neg_idx, :]))
    y = np.concatenate((np.ones(N), np.zeros(N)))
    meta = np.concatenate((pos_meta[pos_idx, :], neg_meta[neg_idx, :]))
    assert N * 2 == X.shape[0] == meta.shape[0] == len(y)
    return X, y, meta


def run_with_cv(classifier, X, y, groups, seed):
    y_tests = []
    y_scores = []
    skf = GroupKFold(n_splits=5)
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y, groups)):
        X_train_fold, X_test_fold = X[train_indices], X[test_indices]
        y_train_fold, y_test_fold = y[train_indices], y[test_indices]

        y_score_fold = classifier(X_train_fold, y_train_fold, X_test_fold, seed=(seed + i))

        y_tests.append(y_test_fold)
        y_scores.append(y_score_fold)

    # Combine test results across folds
    y_test = np.concatenate(y_tests)
    y_score = np.concatenate(y_scores)
    return y_test, y_score


def run_single_method(classifier):
    logging.basicConfig(level='INFO')
    with open('positive_set.json', 'r') as f:
        positive_set, positive_meta = parse_dataset(json.load(f))

    with open('negative_set.json', 'r') as f:
        negative_set, negative_meta = parse_dataset(json.load(f))

    seed = 1
    X, y, meta = get_training_set(positive_set, positive_meta, negative_set, negative_meta, seed)
    groups = meta[:, 1]  # omim id
    y_test, y_score = run_with_cv(classifier, X, y, groups, seed)
    roc_curve(y_test, y_score)


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
