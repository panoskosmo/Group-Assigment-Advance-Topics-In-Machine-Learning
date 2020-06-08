from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

def metrics(X_train, X_test, y_train, y_true, y_pred):
    
    print('Accurancy:')
    acc = accuracy_score(y_true,y_pred)
    print(acc)
    
    
    
    print('Final Classidication Report :')
    clf_rep = classification_report(y_true, y_pred)
    print(clf_rep)
        
    
    print('Multilabel Confusion Matrix:')
    mcm = multilabel_confusion_matrix(y_true,y_pred)
    print(mcm)
    
    print('Precision Recall Curve')
    viz = PrecisionRecallCurve(RandomForestClassifier(n_estimators=10))
    viz.fit(X_train, y_train)
    viz.score(X_test, y_true)
    viz.show()
    
    return