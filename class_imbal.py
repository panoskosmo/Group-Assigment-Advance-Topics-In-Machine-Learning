from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import pandas as pd
from collections import Counter

def class_imbalance(dataframe):
    
    
    print("A quick look at dataframe size:",dataframe.shape)
    
    # print(class_name)
    # c = class_name
    count = dataframe.Class.value_counts()
    minority = 0
    majority = 0
    
    #Check for imbalance

    if count[0]>2*count[1]:
        print('Imbalanced data!! First class has more examples!')
        minority = 1
        print(count)
        print("Minority class is the second (Index:%s):"%minority)

    elif count[1]>2*count[0]:
        print('Imbalanced data!! Second class has more examples!')
        print(count)
        majority = 1
        print("Minority class is the first (Index:%s):"%minority)
    else:
        print('Data are balnced!')
        
        
    #Split to train and test sets with 0.25 test size
    y = dataframe.Class
    x = dataframe.drop('Class', axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)
    
    #Concatenate training data back together
    X = pd.concat([x_train, y_train], axis=1)
    
    
    #Separate minority and majority classes
    minority_df = X[X.Class==minority]
    majority_df = X[X.Class==majority]
    print('Majority len: %g\nMinority len:%g'%(len(majority_df), len(minority_df)))
    
    
    print('\n')
    print(80*'~')
        
    print('1.Random Under-Sampling for majority class',
          '\n2.Random Over-Sampling for minority class',
          '\n3.Over-sampling using SMOTE',
          '\n4.Under-sampling using Tomek Links',
          '\n0.Split to train - test set without any modification.')
 
    print(80*'~')
    print('\n')
    
    try:
        method = int(input("Select a method for deal with imbalanced data:"))
    
    except:
        print('Invalide choice!\nChoose another!')
   
    else:
        #If selected method is 'Random Under-Sampling for majority class'
        if method == 1:
            
            majority_downsampled = resample(majority_df,
                                             replace = False,
                                             n_samples = len(minority_df),
                                             random_state = 27)
            
            
        
            #Combine minority and downsampled majority
            downsampled = pd.concat([majority_downsampled, minority_df])
            print(downsampled.Class.value_counts())
            
            #Reconstruct x_tran, y_train
            y_train = downsampled.Class
            x_train = downsampled.drop('Class', axis=1)
        
        #If selected method is 'Random Over-Sampling for minority class'
        if method == 2:
            
            minority_upsampled = resample(minority_df,
                                         replace = True,
                                         n_samples = len(majority_df),
                                         random_state = 27)
            
            #Combine minority and downsampled majority
            upsampled = pd.concat([minority_upsampled, majority_df])
            print(upsampled.Class.value_counts())
            
            #Reconstruct x_tran, y_train
            y_train = upsampled.Class
            x_train = upsampled.drop('Class', axis=1)
            
        #If selected method is 'Over-sampling using SMOTE'
        if method == 3:
            sm = SMOTE(random_state = 27)         
            x_train, y_train = sm.fit_sample(x_train, y_train)
            
            print('Resampled dataset shape %s' %Counter(y_train))
         
            
        #If selected method is 'Under-sampling using Tomek Links' 
        if method == 4:
            tm = TomekLinks()
            x_train, y_train = tm.fit_sample(x_train, y_train)
            
            print('Resampled dataset shape %s' %Counter(y_train))
            
        #If selected method is 'Split to train - test set without any modification'
        if method == 4:
            print("Return train - test set without any modification")
            
        return(x_train, x_test, y_train, y_test)