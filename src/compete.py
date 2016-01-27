#!/bin/python
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import cross_validation
import pandas as pd
import numpy as np
from time import time
import sys

def output(Ytest,filename):
    out = pd.DataFrame({'Id':1+np.arange(Ytest.size),
        'Prediction':Ytest.astype(int)})
    out.to_csv(filename,index=False)


def Gridsearch_impl(X,Y,clf,param,cv):

    grid_search = GridSearchCV(clf,param,verbose=10,cv=cv,n_jobs=10)
    start = time()
    grid_search.fit(X,Y)
    print(grid_search.grid_scores_)


def PCA_analysis(X, nfeatures):
    pca = PCA(n_components = nfeatures)
    pca.fit(X)
    print(pca.explained_variance_ratio_)



 
def importdata():
    trainf = '../data/training_data.txt'
    testf = '../data/testing_data.txt'
    train_data = np.loadtxt(trainf,delimiter='|',skiprows = 1)
    test_data = np.loadtxt(testf,delimiter='|',skiprows = 1)
    X = train_data[:,1:-1]
    Y = train_data[:,-1]
    N,D = X.shape
    for ii in range(0,D):
        if(np.sum(X[:,ii]) == 0.0):
            print("%d, feature all 0!"%ii)
    Xtest = test_data[:,1:]
    return X,Y,Xtest

def cross_val(X,Y):
    depth = 10
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))
    param_grid={
            "n_estimator":[600]}
    Gridsearch_impl(X,Y,clf,param_grid,cv=5)

def output_trainE(X,Y,clf):
    Ytrain = clf.predict(X)
    print(Ytrain)
    print(Y)
    print(np.sum(np.abs(Y-Ytrain))/Y.shape[0])

def main():

    X,Y,Xtest = importdata()
    print(Y.shape)
    for i in range(10,20):
        clf = DecisionTreeClassifier(min_samples_split=i)
        rf = RandomForestClassifier(n_estimators = 300,random_state=0,min_samples_split=i)
        #ab = AdaBoostClassifier(n_estimators = 100)
        #ab = GradientBoostingClassifier(n_estimators = 100)
        score = cross_validation.cross_val_score(rf,X,Y,cv=3)
        print(score)
        print("average score %f"%np.mean(score))
        print("std %f"%np.std(score))
        rf.fit(X,Y)
   


    Ytest = rf.predict(Xtest)
    output(Ytest,'submit3.csv')


if __name__ == '__main__':
    main()
