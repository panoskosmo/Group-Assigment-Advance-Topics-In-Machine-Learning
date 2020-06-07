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
from skmultilearn.ext import download_meka
from skmultilearn.ext import Meka

##########################################################
######  From multi label to multi class
##########################################################
def multi_labelTo_multi_class (Y, model):
    num_of_labels=Y.ndim
    if (num_of_labels==1):
         print("This is not a multi-label problem!!!!!!")
         return Y
    #LabelPowerset is used here as it contains the transform function
    #that actuall do the multi_label to muti_class transformation.
    transclf = LabelPowerset(classifier=model, require_dense=[False, True])
    return [transclf,transclf.transform(Y)]

##########################################################
######  From multi class to multi label
##########################################################
def multi_classTo_multi_multi(Y, model):
    num_of_labels=Y.ndim
    if (num_of_labels>=2):
         print("This is already a multi-label problem!!!!!!")
         return Y
    transclf = LabelPowerset(classifier=model, require_dense=[False, True])
    return transclf.inverse_transform(Y)


##########################################################
######
##########################################################
def class_multi_label(x, Y, model, wekamodelname, value):

    # detect is the data classification is a multi-label problem.
    num_of_labels=Y.ndim
    print("\n\n-----------------------------------------------------------\n")
    if (num_of_labels==1):
         print("This is not a multi-label problem!!!!!!")
         return model


    while 1:
        if (value<1) or (value>8):
            print("This is a Multi label problem");
            print("Please select:")
            print("1. For binary relevance")
            print("2. For pairwise comparison")
            print("3. Calibrated label ranking")
            print("4. Chain classifier ")
            print("5. PowerSet no pruning ")
            print("6. PowerSet with pruning ")
            print("7. Random-k Labelsets ")
            print("8. Pairwise comparison ")
            value = input("Please enter a choice:\n")

        if value == 1:
            print("Applying binary relevance")
            #clf=BinaryRelevance(classifier=model,require_dense=[False, True])
            if wekamodelname=="nothing":
                print("WEKA does not support this classifier")
                clf=0
                break;
            clf = Meka(
                meka_classifier="meka.classifiers.multilabel.BR",
                weka_classifier=wekamodelname,
                meka_classpath=download_meka(),
                java_command="C:\\""Program Files""\\Java\\jre1.8.0_251\\bin\\java.exe"  # path to java executable
            )
            break;
        elif value == 2:
            print("Duplicates multi-label examples into examples with one label each")
            if wekamodelname=="nothing":
                print("WEKA does not support this classifier")
                clf=0
                break;
            clf = Meka(
                meka_classifier="meka.classifiers.multilabel.RT",
                weka_classifier=wekamodelname,
                meka_classpath=download_meka(),
                java_command="C:\\""Program Files""\\Java\\jre1.8.0_251\\bin\\java.exe"  # path to java executable
            )
            break
        elif value == 3:
            print("Applying calibrated label ranking")
            if wekamodelname=="nothing":
                print("WEKA does not support this classifier")
                clf=0
                break;
            clf = Meka(
                meka_classifier="meka.classifiers.multilabel.MULAN",
                weka_classifier=wekamodelname,
                meka_classpath=download_meka(),
                java_command="C:\\""Program Files""\\Java\\jre1.8.0_251\\bin\\java.exe"  # path to java executable
            )
            break

        elif value == 4:
            print("Applying Chain Classifier")
            clf = ClassifierChain(classifier=model,require_dense=[False, True])
            break
        elif value == 5:
            print("Applying powerset NO pruning")
            clf = LabelPowerset(classifier=model, require_dense=[False, True])
            break;
        elif value == 6:
            print("Applying powerset with pruning")
            if wekamodelname=="nothing":
                print("WEKA does not support this classifier")
                clf=0
                break;
            clf = Meka(
                meka_classifier="meka.classifiers.multilabel.PS",
                weka_classifier=wekamodelname,
                meka_classpath=download_meka(),
                java_command="C:\\""Program Files""\\Java\\jre1.8.0_251\\bin\\java.exe"  # path to java executable
            )
            break
        elif value == 7:
            print("Applying Random-k Labelsets")
            from skmultilearn.ensemble import RakelD
            clf = RakelD(base_classifier=model, base_classifier_require_dense=[False, True],
                         labelset_size=4)
            break
        elif value == 8:
            print("Applying pairwise comparison")
            if wekamodelname=="nothing":
                print("WEKA does not support this classifier")
                clf=0
                break;
            clf = Meka(
                meka_classifier="meka.classifiers.multilabel.LC",
                weka_classifier=wekamodelname,
                meka_classpath=download_meka(),
                java_command="C:\\""Program Files""\\Java\\jre1.8.0_251\\bin\\java.exe"  # path to java executable
            )
            break

        else:
            print("Try again!!!!")
    return clf
    #################################################################
