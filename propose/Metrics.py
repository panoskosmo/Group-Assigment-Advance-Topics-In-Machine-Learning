from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report, hamming_loss
from sklearn.ensemble import RandomForestClassifier
import numpy as np
def metrics(X_train, X_test, y_train, y_true, y_pred, blverbose, labels = None):
    
    print('Accurancy:')
    acc = accuracy_score(y_true,y_pred)
    if blverbose==1:
        print(acc)
       
    
    print('Final Classidication Report:')
    clf_rep = classification_report(y_true, y_pred, labels=labels)
    cflrep = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    if blverbose==1:
        print(clf_rep)
        
    
    print('Multilabel Confusion Matrix:')
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    if blverbose==1:
        print(mcm)

    hamls=hamming_loss(y_true, y_pred)
    if blverbose==1:
        print(hamls)

    
    return(acc,cflrep,mcm, hamls)
