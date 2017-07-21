import numpy as np

##################################################
######## Calculate entropy value #################
##################################################
def entropy(labels):
    """ Computes entropy of 0-1 vector. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)
def change_after_k_seconds(data, k=1):
    data1 = data[0:len(data) -k]
    data2 = data[k:]
    return list(map(lambda x: x[1] - x[0], zip(data1, data2)))
def correct_percentate(a, b):
    return np.mean( a != b ) * 100
