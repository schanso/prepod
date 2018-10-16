import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, LinearSVC
from wyrm.types import Data
from wyrm.processing import lda_train, lda_apply

from prepod.lib.prep import to_feature_vector


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
    batches = {}
    min_len = len(labels)
    for i in range(n_classes):
        curr_ind = np.where(labels == i)
        np.random.shuffle(curr_ind[0])
        if len(curr_ind[0]) < min_len:
            min_len = len(curr_ind[0])
        batches[i] = curr_ind
    for key, val in batches.items():
        batches[key] = val[0][:min_len]
    return np.hstack(tuple(value for key, value in batches.items()))


def train_test_sklearn(data, targets, n_classes, test_size=.3, reshape=True):
    """Splits data into train and test set of equal class proportions

    Params
    ------
        data : ndArray
            data to be split
        targets : list
            target labels
        test_size : float
            proportion of test to training size
        n_classe : int
            number of classes in the data
        reshape : bool
            whether data should be reshaped to two dimensions (necessity
            for sklearn classifiers)

    Returns
    -------
        X_train : ndArray
            training data
        X_test : ndArray
            test data
        y_train : ndArray
            training labels
        y_test : ndArray
            test labels

    --- IN DEVELOPMENT ---
    """
    # TODO: Implement one-hot
    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=test_size, shuffle=True
    )

    ind_train = equalize_proportions(y_train, n_classes=n_classes)
    ind_test = equalize_proportions(y_test, n_classes=n_classes)
    X_train, y_train = X_train[ind_train, :, :], y_train[ind_train]
    X_test, y_test = X_test[ind_test, :, :], y_test[ind_test]

    if reshape:
        n_trials, n_channels, n_samples = X_train.shape
        X_train = X_train.reshape((n_trials, n_channels * n_samples))
        n_trials, n_channels, n_samples = X_test.shape
        X_test = X_test.reshape((n_trials, n_channels * n_samples))

    return X_train, X_test, y_train, y_test


def train_test_wyrm(data, test_size):
    """Splits wyrm Data into train + test set of equal class proportions

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
            data = to_feature_vector(data)
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


def train_test_cv(data, counter=0):
    """"""
    unique_subj = np.unique(data.subj_ids)
    leave_out_subj = unique_subj[counter]
    idx_train = np.where(data.subj_ids != leave_out_subj)
    idx_test = np.where(data.subj_ids == leave_out_subj)
    X_train = data.data[idx_train, :].squeeze()
    X_test = data.data[idx_test, :].squeeze()
    y_train = data.axes[0][idx_train]
    y_test = data.axes[0][idx_test]

    # Equalize proportions in training data
    n_classes = len(np.unique(y_train))
    idx_equalized = equalize_proportions(labels=y_train, n_classes=n_classes)
    X_train = X_train[idx_equalized, :].squeeze()
    y_train = y_train[idx_equalized]

    ax_train = [y_train, data.axes[1]]
    ax_test = [y_test, data.axes[1]]
    names = data.names
    units = data.units
    dat_train = Data(data=X_train, axes=ax_train, names=names, units=units)
    dat_test = Data(data=X_test, axes=ax_test, names=names, units=units)

    return dat_train, dat_test



def lda_vyrm(data_train, data_test, shrink=False):
    """Trains vyrm's LDA classifier and tests it on `data_test`"""
    clf = lda_train(data_train, shrink=shrink)
    out = lda_apply(data_test, clf)
    pred = (np.sign(out) + 1) / 2
    acc = (pred == data_test.axes[0]).sum() / len(pred)
    return acc


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


def svm(data_train, data_test, n_samples=None, kernel='linear'):
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
        clf = LinearSVC()
    else:
        clf = SVC(gamma='auto')

    clf.fit(X, y)
    pred = clf.predict(X_)
    return np.mean(pred == y_)


