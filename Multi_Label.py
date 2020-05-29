from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import pandas as pd
from collections import Counter
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

##########################################################
######  From multi label to multi class
##########################################################
def multi_labelTo_multi_class (Y, model):
    num_of_labels=Y.shape[1]
    if (num_of_labels==1):
         print("This is not a multi-label problem!!!!!!")
         return Y
    #LabelPowerset is used here as it contains the transform function
    #that actuall do the multi_label to muti_class transformation.
    clf = LabelPowerset(classifier=model, require_dense=[False, True])
    return clf.transform(Y)

##########################################################
######  From multi class to multi label
##########################################################
def multi_classTo_multi_multi(Y, model):
    clf = LabelPowerset(classifier=model, require_dense=[False, True])
    return clf.inverse_transform(Y)


##########################################################
######
##########################################################
def class_multi_label(x, Y, model, value):

    # detect is the data classification is a multi-label problem.
    num_of_labels=Y.shape[1]
    print("\n\n-----------------------------------------------------------\n")
    if (num_of_labels==1):
         print("This is not a multi-label problem!!!!!!")
         return model

    while 1:
        if (value<1) or (value>5):
            print("This is a Multi label problem");
            print("Please select:")
            print("1. For binary relevance")
            print("2. For pairwise comparison")
            print("3. Calibrated ranging")
            print("4. PowerSet")
            print("5. Random-k Labelsets")
            value = input("Please enter a choice:\n")

        if value == 1:
            print("Applying binary relevance")
            clf=BinaryRelevance(classifier=model,require_dense=[False, True])
            break;
        elif value == 2:
            print("Applying pairwise comparison")
            clf=OneVsRestClassifier(model)
            break
        elif value == 3:
            print("Applying Chain Classifier")
            clf = ClassifierChain(classifier=model,require_dense=[False, True])
            break
        elif value == 4:
            print("Applying powerset")
            clf = LabelPowerset(classifier=model, require_dense=[False, True])
            break
        elif value == 5:
            print("Applying Random-k Labelsets")
            from skmultilearn.ensemble import RakelD
            clf = RakelD(base_classifier=model,base_classifier_require_dense=[False, True],
                                labelset_size=4)
            break
        else:
            print("Try again!!!!")
    return clf
    #################################################################
