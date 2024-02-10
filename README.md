# README

## Author
Wadood Alam

## Date
10th February 2024

## Assignment
AI 539 Assignment 2: Model Evaluation

## Dependencies / Imports Required

  - Python 
  - NumPy
  - Pandas
  - Scikit-learn
  - train_test_split
  - accuracy_score
  - confusion_matrix
  - cross_val_score
  - KFold
  - StratifiedKFold
  - GroupKFold
  - StratifiedGroupKFold
  - RandomForestClassifier
  - DecisionTreeClassifier
  - KNeighborsClassifier
  - MLPClassifier
  - DummyClassifier


## Instructions

### Program 1: Data Profile

#### Execution 
1. Install the required dependencies using pip: Pandas, NumPy, sklearn
2. Ensure Dataset is contained in the same directory
3. Run the program
4. The data profile will be stored in profile.csv
5. Uncomment the call to Decribe to get a profile on data in 'data_profile.csv'
6. Uncomment the call to Decribe to get a profile on the Activity feature 'activity_profile.csv'

### Program 2: Training and Evaluation 

#### Execution 
1. Install the required dependencies using pip: Pandas, NumPy, sklearn
2. Ensure Dataset is contained in the same directory
3. Run the program using the command `python activity_eval.py`
4. The program will genrate 3 csv files
5. 'dev_accuracy.csv': The output table for generalization estimates on dev set
6. 'held_accuracy.csv': The output table for accuracy scores on with train on dev and test of held-out dataset
7. 'errors.csv': The error(difference) between dev_accuracy and held_accuracy

## Files in the directory 
1. Dataset
2. activity_profile.csv
3. data_profile.csv
4. dev_accuracy.csv
5. held_accuracy.csv
6. errors.csv
7. data_profile.py
8. activity_eval.py
