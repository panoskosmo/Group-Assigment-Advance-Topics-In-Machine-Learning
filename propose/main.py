from class_imbal import *
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

df = pd.read_csv('creditcard.csv')


x_train, x_test, y_train, y_test = class_imbalance(df, class_col_name='Testt')

upsampled = LogisticRegression(solver='liblinear').fit(x_train, y_train)

y_pred = upsampled.predict(x_test)

print("Accurancy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

help(class_imbalance)