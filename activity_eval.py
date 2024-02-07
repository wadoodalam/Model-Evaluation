# TODO: Evaluate four classifiers: DecisionTree, RandomForest, KNeighbors, and MLP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score,StratifiedKFold,GroupKFold,StratifiedGroupKFold 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def LoadData():
    # read from csv(ignore persons col)
    data = pd.read_csv('activity-dev.csv', usecols=['activity','G_front','G_vert','G_lat','ant_id','RSSI','phase','freq','person'])
    # X-col
    X = data[['G_front','G_vert','G_lat','ant_id','RSSI','phase','freq']]
    #Y-cols
    Y = data['activity']
    #Group for group-wise cross validation
    group = data['person']
    return X,Y,group

def DictToCSV(data,col_names):
    df = pd.DataFrame(data)
    df = df.T
    df.columns=col_names
    df.to_csv('dev_accuracy.csv',float_format='%.4f')

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

def TenKFold(X,Y,classifier):
    # init KFold with 10 splits
    k_fold = KFold(n_splits=10,shuffle=True)
    # predict using classifier, with cross-validation as k-folds(10 folds), and find the scores
    scores = cross_val_score(classifier,X,Y,cv=k_fold)
    # take avg of scores
    accuracy = scores.mean()
    return accuracy

def SKFold(X,Y,classifier):
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

def GKFold(X,Y,classifier,group):
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

def SGKFold(X,Y,classifier,group):
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
    DT = DecisionTreeClassifier(random_state=42)
    RF = RandomForestClassifier(random_state=0,min_samples_split=10,n_estimators=100)
    KNN = KNeighborsClassifier(n_neighbors=5) 
    MLP = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,random_state=0)
    # list of classifiers to loop through
    classifiers = {DT:[],RF:[],KNN:[],MLP:[]}
    for classifier,acc in classifiers.items():
        # Call all the methodologies and append to the classifier dict 
        classifiers[classifier].append(TrainTestSplitAccuracy(X,Y,classifier))
        classifiers[classifier].append(TenKFold(X,Y,classifier))
        classifiers[classifier].append(SKFold(X,Y,classifier))
        classifiers[classifier].append(GKFold(X,Y,classifier,group))
        classifiers[classifier].append(SGKFold(X,Y,classifier,group))
    
    col_names = ['A: Train 80%,test 20%(random)', 'B: 10-fold CV','C: Stratified 10-fold CV','D: Group-wise 10-fold CV','E: Stratified Group-wise 10-fold CV']  
    # Convert the the results in dict to a csv file
    DictToCSV(classifiers,col_names)

if __name__ == "__main__":
    
    X,Y,group = LoadData()
    ActivityDevAccuracy(X,Y,group)
        


    