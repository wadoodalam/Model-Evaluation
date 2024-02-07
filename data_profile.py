'''
Author: Wadood Alam
Date: 
Class: AI 539
Assignment: Homework2 - Model Evaluation
''' 
import numpy as np
import pandas as pd

def Describe(data):
    description = data.describe()
    description.to_csv('data_profile.csv')
    
if __name__ == "__main__":
    data = pd.read_csv('activity-dev.csv')
    Describe(data)