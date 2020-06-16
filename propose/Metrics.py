from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report, hamming_loss
import numpy as np


def metrics(X_train, X_test, y_train, y_true, y_pred, blverbose, labels=None):
    acc = accuracy_score(y_true, y_pred)
    if blverbose == 1:
        print('Accurancy:')
        print(acc)
        
    clf_rep = classification_report(y_true, y_pred, labels=labels)
    cflrep = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    if blverbose == 1:
        print('Final Classification Report:')
        print(clf_rep)

    
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    if blverbose == 1:
        print(mcm)

    hamls = hamming_loss(y_true, y_pred)
    if blverbose == 1:
        print('Hamming_loss:')
        print(hamls)
    
    return (acc, cflrep, mcm, hamls)
