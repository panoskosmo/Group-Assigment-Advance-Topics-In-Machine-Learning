from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE

def class_imbal(df,n_feature, transformer, method=''):
    
    '''
        Specifie method arguements as follow (default fuction shows option menu):
              1.Random Over-Sampling (auto),
              2.Random Over-Sampling (not majority),
              3.Random Over-Sampling (all),
              4.Random Over-Sampling using SMOTE (auto),
              5.Random Over-Sampling using SMOTE (not majority),
              6.Random Over-Sampling using SMOTE (all),
              0.Split to train - test set without any modification.
    
    '''
       
    classes = df.Class.unique()
        
    print("Number of Classes:", len(classes))
        
    x = df.iloc[:,0:n_feature] # features
    y = df.Class #classes
    ml_data = df.iloc[:,n_feature:].to_numpy() #keep multi label data
  
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

    print("\n\n-----------------------------------------------------------\n")
    print("Classes before resampling:\n")
    print(sorted(Counter(y_train).items()))
    print("\n\n-----------------------------------------------------------\n")
    
####################################################################################    
###### Menu
####################################################################################
    
    if method == '':
        print('\n')
        print(80*'~')
        
        print('1.Random Over-Sampling (auto)',
              '\n2.Random Over-Sampling (not majority)',
              '\n3.Random Over-Sampling (all)',
              '\n4.Random Over-Sampling using SMOTE (auto)',
              '\n5.Random Over-Sampling using SMOTE (not majority)',
              '\n6.Random Over-Sampling using SMOTE (all)',
              '\n0.Split to train - test set without any modification.')
 
        print(80*'~')
        print('\n')
        
        try:
            method = int(input("Select a method for deal with imbalanced data:"))
    
        except:
            print('Invalide choice!\nChoose another!')
   
   
   #If selected method is 'Random Under-Sampling for majority class'
    if method == 1: 
       
            
        print("\n\n-----------------------------------------------------------\n")
        print('Random Over-Sampling (auto)')
        
        ros = RandomOverSampler(sampling_strategy='auto', random_state=0)        
        x_train, y_train = ros.fit_resample(x_train, y_train)
        x_train=x_train.to_numpy()
        y_train=transformer.inverse_transform(y_train)
    
    if method == 2:  
        
        print("\n\n-----------------------------------------------------------\n")
        print('Random Over-Sampling (not majority)')
        
        ros = RandomOverSampler(sampling_strategy='not majority', random_state=0)        
        x_train, y_train = ros.fit_resample(x_train, y_train)
        x_train=x_train.to_numpy()
        y_train=transformer.inverse_transform(y_train)

    if method == 3:  
        
        print("\n\n-----------------------------------------------------------\n")
        print('Random Over-Sampling (all classes)')
        
        ros = RandomOverSampler(sampling_strategy='all', random_state=0)        
        x_train, y_train = ros.fit_resample(x_train, y_train)
        x_train=x_train.to_numpy()
        y_train=transformer.inverse_transform(y_train)

    if method == 4:  
        
        print("\n\n-----------------------------------------------------------\n")
        print('Random Over-Sampling using SMOTE (auto)')
        
        try:
            sm = SMOTE(sampling_strategy='auto', random_state=0, k_neighbors=2)        
            x_train, y_train = sm.fit_resample(x_train, y_train)
            x_train=x_train.to_numpy()
            y_train=transformer.inverse_transform(y_train)
            
        except ValueError:
               print('\nWarning!!! Not enough instances per class for SMOTE executing!! (Number of neighbors=2)')
               print('Try something else...')            


    if method == 5:  
        
        print("\n\n-----------------------------------------------------------\n")
        print('Random Over-Sampling using SMOTE (not majority)')
        
        try:
            sm = SMOTE(sampling_strategy='not majority', random_state=0, k_neighbors=2)        
            x_train, y_train = sm.fit_resample(x_train, y_train)
            x_train=x_train.to_numpy()
            y_train=transformer.inverse_transform(y_train)
        
        except ValueError:
               print('\nWarning!!! Not enough instances per class for SMOTE executing!! (Number of neighbors=2)')
               print('Try something else...')            


        
    if method == 6:  
        
        print("\n\n-----------------------------------------------------------\n")
        print('Random Over-Sampling using SMOTE (all)')
        
        try:
            sm = SMOTE(sampling_strategy='all', random_state=0, k_neighbors=2)        
            x_train, y_train = sm.fit_resample(x_train, y_train)
            x_train=x_train.to_numpy()
            y_train=transformer.inverse_transform(y_train)

        except ValueError:
                       print('\nWarning!!! Not enough instances per class for SMOTE executing!! (Number of neighbors=2)')
                       print('Try something else...')            

    
    if method == 0:
        print("Return train - test set without any modification")


    #print("Resampled classes:\n")
    #print(sorted(Counter(y_train).items()))
    y_test=transformer.inverse_transform(y_test)
    return(x_train, x_test, y_train, y_test)
