'''
Author: Wadood Alam
Date: 
Class: AI 539
Assignment: Homework2 - Model Evaluation
''' 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score,StratifiedKFold,GroupKFold,StratifiedGroupKFold 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from itertools import chain

def LoadData(path):
    # read from csv(ignore persons col)
    data = pd.read_csv(path, usecols=['activity','G_front','G_vert','G_lat','ant_id','RSSI','phase','freq','person'])
    # X-col
    X = data[['G_front','G_vert','G_lat','ant_id','RSSI','phase','freq']]
    #Y-cols
    Y = data['activity']
    #Group for group-wise cross validation
    group = data['person']
    return X,Y,group


def TrainTestSplitAccuracy(X,Y,classifier):
    # 80% of 3182 = ~2545
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,random_state=0)
    # train classifier
    classifier.fit(X_train,Y_train)
    # predict Y label
    Y_predict = classifier.predict(X_test)
    # find accuracy score
    accuracy = accuracy_score(Y_test,Y_predict)
    return accuracy

def TenKFoldAccuracy(X,Y,classifier):
    # init KFold with 10 splits
    k_fold = KFold(n_splits=10,shuffle=True)
    # predict using classifier, with cross-validation as k-folds(10 folds), and find the scores
    scores = cross_val_score(classifier,X,Y,cv=k_fold)
    # take avg of scores
    accuracy = scores.mean()
    return accuracy

def SKFoldAccuracy(X,Y,classifier):
    sk_fold = StratifiedKFold(n_splits=10,shuffle=True)
    scores = []
    for train,test in sk_fold.split(X,Y):
        # Train and test separation by the sk_fold index
        X_train, X_test = X.iloc[train],X.iloc[test]
        Y_train, Y_test = Y.iloc[train],Y.iloc[test]
        
        # Train the classifier and predict
        classifier.fit(X_train,Y_train)
        Y_predict = classifier.predict(X_test)
        
        # get accuracy scores as numpy
        scores.append(accuracy_score(Y_test,Y_predict))
    # avg of the accuracy scores
    accuracy = np.mean(scores)
    return accuracy

def GKFoldAccuracy(X,Y,classifier,group):
    gk_fold = GroupKFold(n_splits=10)
    scores = []
    for train,test in gk_fold.split(X,Y,group):
        # Train and test separation by the gk_fold index
        X_train, X_test = X.iloc[train],X.iloc[test]
        Y_train, Y_test = Y.iloc[train],Y.iloc[test]
        
        # Train the classifier and predict
        classifier.fit(X_train,Y_train)
        Y_predict = classifier.predict(X_test)
        
        # get accuracy scores as numpy
        scores.append(accuracy_score(Y_test,Y_predict))
    # avg of the accuracy scores
    accuracy = np.mean(scores)
    return accuracy

def SGKFoldAccuracy(X,Y,classifier,group):
    sgk_fold = StratifiedGroupKFold(n_splits=10,shuffle=True)
    scores = []
    for train,test in sgk_fold.split(X,Y,group):
        # Train and test separation by the gk_fold index
        X_train, X_test = X.iloc[train],X.iloc[test]
        Y_train, Y_test = Y.iloc[train],Y.iloc[test]
        
        # Train the classifier and predict
        classifier.fit(X_train,Y_train)
        Y_predict = classifier.predict(X_test)
        
        # get accuracy scores as numpy
        scores.append(accuracy_score(Y_test,Y_predict))
    # avg of the accuracy scores
    accuracy = np.mean(scores)
    return accuracy    

def ActivityDevAccuracy(X,Y,group):
    # init classifiers
    DT = DecisionTreeClassifier(random_state=0)
    RF = RandomForestClassifier(random_state=0,min_samples_split=10,n_estimators=100)
    KNN = KNeighborsClassifier(n_neighbors=3) 
    MLP = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,random_state=0)
    # Dict of classifiers 
    classifiers = {DT:[],RF:[],KNN:[],MLP:[]}
    
    for classifier,acc in classifiers.items():
        # Call all the methodologies and append to the classifier dict 
        classifiers[classifier].append(TrainTestSplitAccuracy(X,Y,classifier))
        classifiers[classifier].append(TenKFoldAccuracy(X,Y,classifier))
        classifiers[classifier].append(SKFoldAccuracy(X,Y,classifier))
        classifiers[classifier].append(GKFoldAccuracy(X,Y,classifier,group))
        classifiers[classifier].append(SGKFoldAccuracy(X,Y,classifier,group))
    
    return classifiers


def TrainModel(X,Y,classifier):
    # train the model
    classifier.fit(X,Y)
    return classifier

def Predict(X,Y,classifier):
    #  predict from the model
    Y_predict = classifier.predict(X)
    # find accuracy
    accuracy = accuracy_score(Y,Y_predict)
    return accuracy

def ActivityHeldOutAccuracy(X,Y,X_held,Y_held):
    # TODO: Baseline Algorithm
    # Train each model with dev dataset
    DT = TrainModel(X,Y,DecisionTreeClassifier(random_state=0))
    RF = TrainModel(X,Y,RandomForestClassifier(random_state=0,min_samples_split=10,n_estimators=100))
    KNN = TrainModel(X,Y,KNeighborsClassifier(n_neighbors=5)) 
    MLP = TrainModel(X,Y,MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,random_state=0))   
    Dummy = TrainModel(X,Y,DummyClassifier(strategy='stratified',random_state=0)) 
    
    # Predict with each model and get accuracy in comparison to held-out dataset
    DT_accuracy = Predict(X_held,Y_held,DT)
    RF_accuracy = Predict(X_held,Y_held,RF)
    KNN_accuracy = Predict(X_held,Y_held,KNN)
    MLP_accuracy = Predict(X_held,Y_held,MLP)
    Dummy_accuracy = Predict(X_held,Y_held,Dummy)
    
    classifiers_accuracy = {DT:[DT_accuracy],RF:[RF_accuracy],KNN:[KNN_accuracy],MLP:[MLP_accuracy],Dummy:[Dummy_accuracy]}
    
    return classifiers_accuracy
    

   
    
if __name__ == "__main__":
    col_names = ['A: Train 80%,test 20%(random)', 'B: 10-fold CV','C: Stratified 10-fold CV','D: Group-wise 10-fold CV','E: Stratified Group-wise 10-fold CV']  
    X,Y,group = LoadData('activity-dev.csv')
    X_held,Y_held,group_held = LoadData('activity-heldout.csv')
    
    activity_dev_accuracy = ActivityDevAccuracy(X,Y,group)
    activity_held_accuracy = ActivityHeldOutAccuracy(X,Y,X_held,Y_held)   




    held_acc = []
    for classifier, acc in activity_held_accuracy.items():
        for accuracy in acc:
            held_acc.append(accuracy)
    held_acc = held_acc[:-1]

 
    dev_acc = []
    for classifier, acc in activity_dev_accuracy.items():
        dev_acc.append(acc)
    #print("Dev", dev_acc)
    #print("Held", held_acc)
    error = []
    j = -1
    for element in dev_acc:
        j+=1
        signed_error = []
        for i in element:
            signed_error.append(abs(i-held_acc[j]))
        error.append(signed_error)
    
    
            
    # Create dict for csv output for error
    keys = ['Decision Tree', 'Random Forest', '3-NN', 'MLP']
    errors = {}
    for i, j in enumerate(error):
        key = keys[i]  
        errors[key] = j

    
 
 
 
    # Convert the the results in dict to a csv file
    if activity_dev_accuracy:
        df = pd.DataFrame(activity_dev_accuracy)
        df = df.T
        df.columns=col_names 
        df.to_csv('dev_accuracy.csv')

    if activity_held_accuracy:
        df = pd.DataFrame(activity_held_accuracy)
        df = df.T
        df.columns=['Held-out Accuracy'] 
        df.to_csv('held_accuracy.csv')
        
    if errors:
        df = pd.DataFrame(errors)
        df = df.T
        averages = df.mean()
        df.loc['Average'] = averages    
        df.columns=col_names 
        df.to_csv('errors.csv')

    