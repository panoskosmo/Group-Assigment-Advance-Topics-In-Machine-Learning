from class_imbal import *
from Multi_Label import *
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.calibration import CalibratedClassifierCV


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
#this is a numpy array carrying the multi_label classes
Y=df.to_numpy()

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
upsampled = LogisticRegression(solver='liblinear')

###############################################
# convert Y from multilabel to multi class
yt=multi_labelTo_multi_class(df.to_numpy(),upsampled)

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

df.insert(loc=len(df.columns),column='Class',value=yt)
print(df)

##############################################
#df = pd.read_csv('creditcard.csv')

#### WARNING  the returned x_train and x_test must exclude the multi_label info
#### and as output return the multi_label columns and NOT the multi_class column
x_train, x_test, y_train, y_test, ml_data = class_imbal(df, 5)


#################################################################################################################
#Pano thelei apo edo kai kato douleia kai alli aplos kane to imbalance gia na mporeso na kano to debug

###############################################
##### detect cost sensitivity and act with weighted sampling

##############################################
#it is required a base algorithm callibration before any cost sensitivity action
upsampled=CalibratedClassifierCV(base_estimator=upsampled, method='sigmoid', cv=None)
fclf=""
fclf[0] = class_multi_label(x_train, y_train,upsampled,1)
fclf[1] = class_multi_label(x_train, y_train,upsampled,2)
fclf[2] = class_multi_label(x_train, y_train,upsampled,3)
fclf[3] = class_multi_label(x_train, y_train,upsampled,4)
fclf[4] = class_multi_label(x_train, y_train,upsampled,5)
upsampled=CalibratedClassifierCV(base_estimator=upsampled, method='isotonic', cv=None)
fclf[5] = class_multi_label(x_train, y_train,upsampled,1)
fclf[6] = class_multi_label(x_train, y_train,upsampled,2)
fclf[7] = class_multi_label(x_train, y_train,upsampled,3)
fclf[8] = class_multi_label(x_train, y_train,upsampled,4)
fclf[9] = class_multi_label(x_train, y_train,upsampled,5)

#Pano thelei kai alli douleia edo aplos kane to imbalance gia na mporeso na kano debug edo
for clf in fclf:
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    #print(classification_report(y_test, y_pred, target_names=data.target_names))
    #loss = cost_loss(y_test, y_pred, cost_matrix)
    #print(confusion_matrix(y_test, y_pred).T

    print("Accurancy:", accuracy_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))

    help(class_imbalance)