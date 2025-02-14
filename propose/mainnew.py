import warnings
warnings.filterwarnings("ignore")

import sys
from class_imbal import *
from Multi_Label import *
from Metrics import *
#from interpret import *
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from skmultilearn.ext import download_meka
from skmultilearn.ext import Meka
from ipywidgets import interactive
from costcla.metrics import cost_loss

#### reading artff files of a gene instance file.
data = arff.loadarff('genbase.arff')
df = pd.DataFrame(data[0])
print(df)
rr = df.iloc[:, 0].to_numpy()
yy=np.empty([rr.shape[0],6],dtype=np.uint8)
i=0
for lclrr in rr:
    yy[i]=np.fromstring(lclrr, dtype=np.uint8)
    i=i+1

df=df.drop('protein',axis=1)
for name in df.columns:
    print("\nProcessed:",name)
    rr= df.loc[:,name].to_numpy()
    tt=np.where(((rr==b'YES') | (rr==b'1')),1,0)
    df.loc[:,name]=tt


##########################################
### this are the feature instances
### from column 5> are the multi_label classes
df.insert(loc=0,column="ft4",value=yy[:,5])
df.insert(loc=0,column="ft3",value=yy[:,4])
df.insert(loc=0,column="ft2",value=yy[:,3])
df.insert(loc=0,column="ft1",value=yy[:,2])
df.insert(loc=0,column="ft0",value=yy[:,1])

###############################################
###########  get basic model
arclfs = [
          [RandomForestClassifier(n_estimators=100, random_state=0),"weka.classifiers.trees.RandomForest"],
          [SVC(kernel='linear', probability=True, C=1), "weka.classifiers.functions.SMO"]
         ]
choosenclsfr=0
###############################################
# convert Y from multilabel to multi class
transformer, yt=multi_labelTo_multi_class(df.to_numpy(),arclfs[choosenclsfr][0])

##################################################################################
df.insert(loc=len(df.columns),column='Class',value=yt)

###############################################
##### detect cost sensitivity and act with weighted sampling
print("The Labels are:\n")
print(df.iloc[:,6:])
costclass= int(input("\n\n\nWhich label is cost sensitive type the class number to over sample it:"))
costval= int(input("\nWhat is the cost?(integer):"))

#perfom the imbalancing actions
for ampl_m in range(1,6):
    print("\n==================================================\n---------------------------------------New sampling method\n")
    x_train, x_test, y_train, y_test = class_imbal(df, 5,transformer,costclass,costval,ampl_m)

    ##############################################
    #it is required a base algorithm callibration before any cost sensitivity action
    upsampled=CalibratedClassifierCV(base_estimator=arclfs[0][0], method='sigmoid', cv=None)
    upsampled2=CalibratedClassifierCV(base_estimator=arclfs[1][0], method='isotonic', cv=None)
    fclf = [
            #[class_multi_label(x_train, y_train, 0, 0, 9), "Applying Multilabel k Nearest Neighbours", "SVM_KK - MLKnn"],
            [class_multi_label(x_train, y_train,upsampled,arclfs[0][1],1),"Applying binary relevance", "RFC - Binary Relevance"],
            [class_multi_label(x_train, y_train,upsampled,arclfs[0][1],2),"Duplicates multi-label examples into examples with one label each", "RFC - Multi-label examples into examples with one label each"],
            [class_multi_label(x_train, y_train,upsampled,arclfs[0][1],3),"Applying calibrated label ranking", "RFC - Calibrated label ranking"],
            [class_multi_label(x_train, y_train,upsampled,arclfs[0][1],4),"Applying Chain Classifier", "RFC - Chain Classifier"],
            [class_multi_label(x_train, y_train,upsampled,arclfs[0][1],5),"Applying powerset NO pruning", "RFC - Powerset NO pruning"],
            [class_multi_label(x_train, y_train,upsampled,arclfs[0][1],6),"Applying powerset with pruning", "RFC - Powerset with pruning"],
            [class_multi_label(x_train, y_train,upsampled,arclfs[0][1],7),"Applying Random-k Labelsets", "RFC - Random-k Labelsets"],
            [class_multi_label(x_train, y_train,upsampled,arclfs[0][1],8),"Applying pairwise comparison", "RFC - Pairwise comparison"],
            [class_multi_label(x_train, y_train,upsampled2,arclfs[1][1],1),"Applying binary relevance", "SVC - Binary Relevance"],
            [class_multi_label(x_train, y_train,upsampled2,arclfs[1][1],2),"Duplicates multi-label examples into examples with one label each", "SVC - Multi-label examples into examples with one label each"],
            [class_multi_label(x_train, y_train,upsampled2,arclfs[1][1],3),"Applying calibrated label ranking", "SVC - Calibrated label ranking"],
            [class_multi_label(x_train, y_train,upsampled2,arclfs[1][1],4),"Applying Chain Classifier", "SVC - Chain Classifier"],
            [class_multi_label(x_train, y_train,upsampled2,arclfs[1][1],5),"Applying powerset NO pruning", "SVC - Powerset NO pruning"],
            [class_multi_label(x_train, y_train,upsampled2,arclfs[1][1],6),"Applying powerset with pruning", "SVC - Powerset with pruning"],
            [class_multi_label(x_train, y_train,upsampled2,arclfs[1][1],7),"Applying Random-k Labelsets", "SVC - Random-k Labelsets"],
            [class_multi_label(x_train, y_train,upsampled2,arclfs[1][1],8),"Applying pairwise comparison", "SVC - Pairwise comparison"]
           ]

    print("\n\nEntering main fitting and predicting loop\n\n")

    all_metrics = {}

    for clf in fclf:
        print("Fitting-------->>>>>>>>  ",clf[1])
        if clf[0]==0:
            continue;
        clf[0].fit(x_train, y_train)
        y_pred = clf[0].predict(x_test)
        print("Some evaluation metrics for classifier:\n")
        acc,cflrep,mcm, hamls = metrics(x_train, x_test, y_train, y_test, y_pred, 0)#, labels=df.Class)
        #########################
        # convert Y from multilabel to multi class
        b_y_test = multi_labelTo_multi_class_D(y_test, transformer)
        b_y_pred = multi_labelTo_multi_class_D(y_pred, transformer)
        b_y_test=np.where(np.array(b_y_test)==costclass,1,0)
        b_y_pred=np.where(np.array(b_y_pred)==costclass,1,0)
        fp = np.full((b_y_test.shape[0], 1), costval)
        fn = np.full((b_y_test.shape[0], 1), 1)
        tp = np.zeros((b_y_test.shape[0], 1))
        tn = np.zeros((b_y_test.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))
        loss = cost_loss(b_y_test, b_y_pred, cost_matrix)
        ###########################
        all_metrics.update({clf[2]:{'Accurancy':acc, 'Classification Report':cflrep, 'Confusion Matrix':mcm,'Hamming Loss':hamls,'Cost Loss':loss}})


    print('\n\n=============================================================================\n')
    print('---------------Final Results-----------------')

    for key in all_metrics.keys():
        print(key)
        print("\n")
        print("Accurancy: %g%%"%round(all_metrics[key]['Accurancy']*100,2))
        print("Hamming LossAccurancy: %g%%"%round(all_metrics[key]['Hamming Loss']*100,2))
        print("misclassification cost: %g%%"%round(all_metrics[key]['Cost Loss']*100,2))
        print("Micro averaging: Precision: %g%% | Recall: %g%% | F1 Score: %g%% | Support: %g"%(round(all_metrics[key]['Classification Report']['micro avg']['precision']*100,2), round(all_metrics[key]['Classification Report']['micro avg']['recall']*100,2), round(all_metrics[key]['Classification Report']['micro avg']['f1-score']*100,2), all_metrics[key]['Classification Report']['micro avg']['support']))
        print("Macro averaging: Precision: %g%% | Recall: %g%% | F1 Score: %g%% | Support: %g"%(round(all_metrics[key]['Classification Report']['macro avg']['precision']*100,2), round(all_metrics[key]['Classification Report']['macro avg']['recall']*100,2), round(all_metrics[key]['Classification Report']['macro avg']['f1-score']*100,2), all_metrics[key]['Classification Report']['macro avg']['support']))
        print("Samples averaging: Precision: %g%% | Recall: %g%% | F1 Score: %g%% | Support: %g"%(round(all_metrics[key]['Classification Report']['samples avg']['precision']*100,2), round(all_metrics[key]['Classification Report']['samples avg']['recall']*100,2), round(all_metrics[key]['Classification Report']['samples avg']['f1-score']*100,2), all_metrics[key]['Classification Report']['samples avg']['support']))
        print("Weighted averaging: Precision: %g%% | Recall: %g%% | F1 Score: %g%% | Support: %g"%(round(all_metrics[key]['Classification Report']['weighted avg']['precision']*100,2), round(all_metrics[key]['Classification Report']['weighted avg']['recall']*100,2), round(all_metrics[key]['Classification Report']['weighted avg']['f1-score']*100,2), all_metrics[key]['Classification Report']['weighted avg']['support']))
        print('\n-------------------------------------------------------------------------------\n')
        with open('results.txt', 'a') as f:
            print(key,file=f)
            print("\n",file=f)
            print("Accurancy: %g%%" % round(all_metrics[key]['Accurancy'] * 100, 2),file=f)
            print("Hamming LossAccurancy: %g%%" % round(all_metrics[key]['Hamming Loss'] * 100, 2),file=f)
            print("Micro averaging: Precision: %g%% | Recall: %g%% | F1 Score: %g%% | Support: %g" % (
                round(all_metrics[key]['Classification Report']['micro avg']['precision'] * 100, 2),
                round(all_metrics[key]['Classification Report']['micro avg']['recall'] * 100, 2),
                round(all_metrics[key]['Classification Report']['micro avg']['f1-score'] * 100, 2),
                all_metrics[key]['Classification Report']['micro avg']['support']),file=f)
            print("Macro averaging: Precision: %g%% | Recall: %g%% | F1 Score: %g%% | Support: %g" % (
                round(all_metrics[key]['Classification Report']['macro avg']['precision'] * 100, 2),
                round(all_metrics[key]['Classification Report']['macro avg']['recall'] * 100, 2),
                round(all_metrics[key]['Classification Report']['macro avg']['f1-score'] * 100, 2),
                all_metrics[key]['Classification Report']['macro avg']['support']),file=f)
            print("Samples averaging: Precision: %g%% | Recall: %g%% | F1 Score: %g%% | Support: %g" % (
                round(all_metrics[key]['Classification Report']['samples avg']['precision'] * 100, 2),
                round(all_metrics[key]['Classification Report']['samples avg']['recall'] * 100, 2),
                round(all_metrics[key]['Classification Report']['samples avg']['f1-score'] * 100, 2),
                all_metrics[key]['Classification Report']['samples avg']['support']),file=f)
            print("Weighted averaging: Precision: %g%% | Recall: %g%% | F1 Score: %g%% | Support: %g" % (
                round(all_metrics[key]['Classification Report']['weighted avg']['precision'] * 100, 2),
                round(all_metrics[key]['Classification Report']['weighted avg']['recall'] * 100, 2),
                round(all_metrics[key]['Classification Report']['weighted avg']['f1-score'] * 100, 2),
                all_metrics[key]['Classification Report']['weighted avg']['support']),file=f)
            print('\n-------------------------------------------------------------------------------\n',file=f)
            f.close()
