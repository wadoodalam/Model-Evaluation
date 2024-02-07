# TODO: Evaluate four classifiers: DecisionTree, RandomForest, KNeighbors, and MLP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score,StratifiedKFold 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def LoadData():
    # read from csv(ignore persons col)
    data = pd.read_csv('activity-dev.csv', usecols=['activity','G_front','G_vert','G_lat','ant_id','RSSI','phase','freq'])
    # X-col
    X = data[['G_front','G_vert','G_lat','ant_id','RSSI','phase','freq']]
    #Y-cols
    Y = data['activity']
  
    return X,Y

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

def CrossValidation(X,Y,classifier):
    # init KFold with 10 splits
    k_fold = KFold(n_splits=10)
    # predict using classifier, with cross-validation as k-folds(10 folds), and find the scores
    scores = cross_val_score(classifier,X,Y,cv=k_fold)
    # take avg of scores
    accuracy = scores.mean()
    return accuracy



if __name__ == "__main__":
    
    X,Y = LoadData()
    # init classifiers
    DT = DecisionTreeClassifier(random_state=42)
    RF = RandomForestClassifier(random_state=0,min_samples_split=10,n_estimators=100)
    KNN = KNeighborsClassifier(n_neighbors=5) 
    MLP = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,random_state=0)
    # list of classifiers to loop through
    classifiers = {DT:[],RF:[],KNN:[],MLP:[]}
    for classifier,acc in classifiers.items():
        classifiers[classifier].append(TrainTestSplitAccuracy(X,Y,classifier))
        classifiers[classifier].append(CrossValidation(X,Y,classifier))
        
    print(classifiers)
        


    