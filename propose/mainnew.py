from class_imbal import *
from Multi_Label import *
from Metrics import *
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


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
#upsampled = LogisticRegression(solver='liblinear')
upsampled = RandomForestClassifier(n_estimators=100)

###############################################
# convert Y from multilabel to multi class
transformer, yt=multi_labelTo_multi_class(df.to_numpy(),upsampled)

##################################################################################
#WARNING we add at the end of dataframe a column 'Class'
#with the multi-class values of multilabel classes found
#on the original df dataframe.
#We do not drop the multi_label data column as they will be used
#by the final multi_label classificator
#Any imbalance algorithm will take into account the 5 first feature
#columns and #the last multi-class column and not the multi-label columns,
#but the multi_label columns must still exist and copied furing imbalance
#algorithms as to be used later on the final multi-lable classificator.
##################################################################################
#unique,count= np.unique(yt, return_counts=True)

df.insert(loc=len(df.columns),column='Class',value=yt)
print(df)

###############################################
##### detect cost sensitivity and act with weighted sampling

#### WARNING  the returned x_train and x_test must exclude the multi_label info
#### and as output return the multi_label columns and NOT the multi_class column
x_train, x_test, y_train, y_test = class_imbal(df, 5,transformer)

##############################################
#it is required a base algorithm callibration before any cost sensitivity action
upsampled=CalibratedClassifierCV(base_estimator=upsampled, method='sigmoid', cv=None)
upsampled2=CalibratedClassifierCV(base_estimator=upsampled, method='isotonic', cv=None)
fclf = {class_multi_label(x_train, y_train,upsampled,1),
        class_multi_label(x_train, y_train,upsampled,2),
        class_multi_label(x_train, y_train,upsampled,3),
        class_multi_label(x_train, y_train,upsampled,4),
        class_multi_label(x_train, y_train,upsampled,5),
        class_multi_label(x_train, y_train,upsampled2,1),
        class_multi_label(x_train, y_train,upsampled2,2),
        class_multi_label(x_train, y_train,upsampled2,3),
        class_multi_label(x_train, y_train,upsampled2,4),
        class_multi_label(x_train, y_train,upsampled2,5)}

for clf in fclf:
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    #print(classification_report(y_test, y_pred, target_names=data.target_names))
    #loss = cost_loss(y_test, y_pred, cost_matrix)
    #print(confusion_matrix(y_test, y_pred).T

    print("Accurancy:", accuracy_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))

    #help(class_imbalance)