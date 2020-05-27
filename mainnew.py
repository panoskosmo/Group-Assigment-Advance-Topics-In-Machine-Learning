from class_imbal import *
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score


#### reading artff files
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
    #for arta in rr:
    #    if arta==b'YES' :
    #      print("Found at",name)
    #print(np.any(rr == b'NO'))
    tt=np.where(((rr==b'YES') | (rr==b'1')),1,0)
    #print(np.any(tt == 1))
    df.loc[:,name]=tt
    #print(df)

df.insert(loc=0,column="ft4",value=yy[:,5])
df.insert(loc=0,column="ft3",value=yy[:,4])
df.insert(loc=0,column="ft2",value=yy[:,3])
df.insert(loc=0,column="ft1",value=yy[:,2])
df.insert(loc=0,column="ft0",value=yy[:,1])



#df = pd.read_csv('creditcard.csv')

ttt=df.head()
print(df)

x_train, x_test, y_train, y_test = class_imbalance(df, class_col_name='Class')

upsampled = LogisticRegression(solver='liblinear').fit(x_train, y_train)

y_pred = upsampled.predict(x_test)

print("Accurancy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

help(class_imbalance)