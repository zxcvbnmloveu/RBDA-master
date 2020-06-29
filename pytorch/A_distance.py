import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn import metrics
from sklearn import svm

def proxy_a_distance(source_X, target_X):

    """
    Compute the Proxy-A-Distance of a source/target representation

    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]
    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int), np.ones(nb_target, dtype=int)))
    # clf = svm.LinearSVC(random_state=0)
    clf = svm.SVC(gamma = 'auto')
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist

def proxy_a_distance2(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    clf = svm.SVC(kernel='linear', verbose=False)
    clf.fit(train_X, train_Y)

    train_risk = np.mean(clf.predict(train_X) != train_Y)
    test_risk = np.mean(clf.predict(test_X) != test_Y)

    if test_risk > .5:
        test_risk = 1. - test_risk

    return 2 * (1. - 2 * test_risk)

def proxy_a_distance3(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 5, 100)

    half_source, half_target = int(nb_source/4), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)

if __name__ == '__main__':
    name_list = ['64DAAOT','64DACDAN','64DARES','64WAAOT','64WACDAN','64WARES']
    for name in name_list:
        source_X = np.load('./snapshot/model/'+name+'_s.npy')
        target_X = np.load('./snapshot/model/'+name+'_t.npy')
        dis = proxy_a_distance3(source_X, target_X)
        print(name + str(dis))
