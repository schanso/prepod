import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, LinearSVC
from wyrm.types import Data

import prepod.lib.prep as prep


def encode_one_hot(targets):
    """Encodes list of targets as one-hot

    Params
    ------
        targets : list
            list of target labels

    Returns
    -------
        onehot_targets : ndArray
            array of one-hot encoded target labels

    -- IN DEVELOPMENT --
    """
    # TODO: Implement
    classes = 2
    return np.eye(classes)[targets].reshape(targets.shape[0], classes)


def equalize_proportions(labels, n_classes):
    """Returns indices from labels with equal class proportions

    Params
    ------
        labels : list
            list of labels for a given dataset
        n_classes : int
            number of classes in the dataset

    Returns
    -------
        indices : list
            list of shuffled indices with equal class proportions

    """
    labels = [str(el) for el in labels]
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    batches = {}
    min_len = len(labels)
    for i in range(n_classes):
        curr_ind = np.where(labels == str(i))
        np.random.shuffle(curr_ind[0])
        if len(curr_ind[0]) < min_len:
            min_len = len(curr_ind[0])
        batches[i] = curr_ind
    for key, val in batches.items():
        batches[key] = val[0][:min_len]
    return np.hstack(tuple(value for key, value in batches.items()))


def train_test(data, test_size):
    """Splits data into train and test set of equal class proportions

    Params
    ------
        data : wyrm.Data
            data to be split
        test_size : float
            between 0 and 1, proportion of test to training size

    Returns
    -------
        dat_train : wyrm.Data
            Data object holding training data
        dat_test : wyrm.Data
            Data object holding test data

    Raises
    ------
        TypeError if data of wrong shape
    """
    labels = data.axes[0]
    n_classes = len(np.unique(labels))
    ind = equalize_proportions(labels=labels, n_classes=n_classes)
    if len(data.axes) > 2:
        try:
            data = prep.create_fvs(data)
        except Exception as e:
            msg = ('It seems you have to reshape your data first.\n\n'
                   + str(e))
            raise TypeError(msg)
    dat = data.data[ind, :]
    labels = data.axes[0][ind]
    X_train, X_test, y_train, y_test = train_test_split(
        dat, labels, test_size=test_size, shuffle=True
    )
    ax_train = [y_train, data.axes[1]]
    ax_test = [y_test, data.axes[1]]
    names = data.names
    units = data.units
    dat_train = Data(data=X_train, axes=ax_train, names=names, units=units)
    dat_test = Data(data=X_test, axes=ax_test, names=names, units=units)

    return dat_train, dat_test


def train_test_cv(data, n_leave_out=1, idx=0):
    """Splits data into train and test according to LOOCV

    Leave-one-subject-out cross validation where n-1 subjects are used
    for training, subject(n) for testing. The training set is subset to
    yield equal class proportions.

    Params
    ------
        data : wyrm.types.Data
            data to be split into train and test set
        leave_out : int
            subj_id in unique subject at index `leave_out` is left out
            of the training set

    Returns
    -------
        dat_train : wyrm.Data
            Data object holding training data
        dat_test : wyrm.Data
            Data object holding test data
    """
    # Subjects to exclude
    unique_subj = np.unique(data.subj_id)
    leave_out_subj = [unique_subj[idx]]

    if n_leave_out > 1:
        info = list(set(list(zip(data.subj_id, data.axes[0]))))
        subj_label = [el[1] for el in info if el[0] == leave_out_subj[0]][0]
        combine_with = [el[0] for el in info
                        if el[1] != subj_label and el[0] != leave_out_subj[0]]
        np.random.shuffle(combine_with)
        leave_out_subj.append(combine_with[0])

    # Subset data
    idx_train = np.where(~np.isin(data.subj_id, leave_out_subj))
    idx_test = np.where(np.isin(data.subj_id, leave_out_subj))
    X_train = data.data[idx_train, :].squeeze()
    X_test = data.data[idx_test, :].squeeze()
    y_train = data.axes[0][idx_train]
    y_test = data.axes[0][idx_test]

    # Equalize proportions in training data
    n_classes = len(np.unique(y_train))
    idx_equalized = equalize_proportions(labels=y_train, n_classes=n_classes)
    X_train = X_train[idx_equalized, :].squeeze()
    y_train = y_train[idx_equalized]

    # Equalize proportions in test data
    n_classes = len(np.unique(y_test))
    idx_equalized = equalize_proportions(labels=y_test, n_classes=n_classes)
    X_test = X_test[idx_equalized, :].squeeze()
    y_test = y_test[idx_equalized]

    ax_train = data.axes[:]
    ax_train[0] = y_train
    ax_test = data.axes[:]
    ax_test[0] = y_test
    names = data.names[:]
    units = data.units[:]
    dat_train = data.copy(data=X_train, axes=ax_train, names=names, units=units)
    dat_test = data.copy(data=X_test, axes=ax_test, names=names, units=units)

    return dat_train, dat_test, leave_out_subj


def train_test_info(X, y, test_size=.3):
    """Splits features + target into train and test set (equal props)"""
    # define train, test indices
    idx = np.arange(X.shape[0])
    n_test = np.ceil(idx.shape[0] * test_size).astype(int)
    n_train = idx.shape[0] - n_test
    np.random.shuffle(idx)
    idx_train = idx[:n_train]
    idx_test = idx[n_train:]

    # split into train and test
    X_train = np.array(X[idx_train])
    X_test = np.array(X[idx_test])
    y_train = np.array(y[idx_train])
    y_test = np.array(y[idx_test])

    # Equalize proportions in training data
    n_classes = len(np.unique(y_train))
    idx_equalized = equalize_proportions(labels=y_train, n_classes=n_classes)
    X_train = X_train[idx_equalized, :]
    y_train = y_train[idx_equalized]

    # Equalize proportions in test data
    n_classes = len(np.unique(y_test))
    idx_equalized = equalize_proportions(labels=y_test, n_classes=n_classes)
    X_test = X_test[idx_equalized, :]
    y_test = y_test[idx_equalized]

    return X_train, X_test, y_train, y_test


def lda(data_train, data_test, solver='lsqr', shrinkage=True):
    """Trains and test LDA classifier"""
    if solver not in ['svd', 'lsqr', 'eigen']:
        msg = 'Solver must be one of \'svd\', \'lsqr\', \'eigen\'.'
        raise ValueError(msg)
    if solver == 'svd' and shrinkage == True:
        msg = 'Shrinkage can only be used with \'lsqr\' and \'eigen\' solvers.'
        raise ValueError(msg)

    X, y = data_train.data, data_train.axes[0]
    X_, y_ = data_test.data, data_test.axes[0]
    clf = LDA(solver=solver)
    clf.fit(X, y)
    pred = clf.predict(X_)
    return np.mean(pred == y_)


def svm(data_train, data_test, n_samples=None, kernel='linear', max_iter=5000):
    """Trains and tests SVC

    --- IN DEVELOPMENT ---
    """
    X, y = data_train.data, data_train.axes[0]
    X_, y_ = data_test.data, data_test.axes[0]

    if n_samples:
        if not isinstance(n_samples, int):
            msg = 'n_samples must be int, got {}'.format(type(n_samples))
            raise TypeError(msg)
        X, y = X[:n_samples], y[:n_samples]
        X_, y_ = X_[:n_samples], y_[:n_samples]

    if kernel == 'linear':
        clf = LinearSVC(max_iter=max_iter)
    else:
        clf = SVC(gamma='auto')

    clf.fit(X, y)
    pred = clf.predict(X_)
    return np.mean(pred == y_)


def process_subset(X, y, clf, n_iterations=20):
    """"""
    acc = []
    for _ in np.arange(n_iterations):
        X_train, X_test, y_train, y_test = train_test_info(X, y)
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        acc.append(score)
    return {'mean_acc': np.mean(acc),
            'std': np.std(acc),
            'iterations': n_iterations,
            'clf': clf}


def forward_subset_selection(data, K, init_combos=3, **kwargs):
    """"""
    results = []
    curr_best = np.array([])
    for k in range(init_combos, K+1):
        # Init with all possible N-fold combinations, then add to the best
        current_labels = np.setdiff1d(data['X_labels'], curr_best)
        if k == init_combos:
            combos = itertools.combinations(current_labels, k)
            n_combos = sum(1 for _ in combos)
            combos = itertools.combinations(current_labels, k)
        else:
            combos = (tuple(np.append(curr_best, el)) for el in current_labels)
            n_combos = len(current_labels)

        # Train and test on each combo
        for i, combo in enumerate(combos):
            print('{}/{}'.format(i + 1, n_combos))
            idx = np.where(np.in1d(data['X_labels'], combo))
            X = data['X'][:, idx].squeeze()
            if k == 1:
                X = X.reshape(-1, 1)
            res = process_subset(X, data['y'], kwargs['clf'])
            res.update({'features': combo, 'n_features':k})
            results.append(res)

        # Pass current best to use as base in next iteration
        df = pd.DataFrame(results)
        df = df[df['n_features'] == k]
        df = df.sort_values(by='mean_acc', ascending=False)
        df.reset_index(inplace=True)
        curr_best = np.array(df.loc[0, 'features'])

    return results


def backward_subset_selection(X, y, X_labels, y_labels, K=1, **kwargs):
    """"""
    if not isinstance(y_labels, np.ndarray) or y_labels.ndim == 0:
        y_labels = np.array(y_labels).reshape(1)
    data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    columns = np.concatenate((X_labels, y_labels), axis=0)
    df = pd.DataFrame(data=data, columns=columns)

    features = list(X_labels)
    updated_features = features.copy()
    n_features = len(features)

    counter = n_features+1
    results = []
    while len(updated_features) > K:
        counter -= 1
        if counter == n_features:
            combos = [features]
            n_combos = len(combos)
        else:
            combos = itertools.combinations(updated_features,
                                            len(updated_features)-1)
            n_combos = len([el for el in combos])
            combos = itertools.combinations(updated_features,
                                            len(updated_features)-1)

        for i, combo in enumerate(combos):
            print('{}/{} for k = {}'.format(i + 1, n_combos, counter))
            dropped = ''.join([el for el in updated_features if el not in combo])
            curr_X = df[list(combo)].values
            res = process_subset(curr_X, y, kwargs['clf'])
            res.update({'features': combo,
                        'n_features': len(combo),
                        'dropped': dropped})
            results.append(res)

        # Pass current best to use as base in next iteration
        res_df = pd.DataFrame(results)
        res_df = res_df[res_df['n_features'] == counter]
        res_df = res_df.sort_values(by='mean_acc', ascending=False)
        res_df.reset_index(inplace=True)
        dropped_in_best = res_df.loc[0, 'dropped']
        try:
            updated_features.remove(dropped_in_best)
        except ValueError:  # nothing dropped in first
            pass

    return results

