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
from operator import itemgetter

def output(Ytest,filename):
    out = pd.DataFrame({'Id':1+np.arange(Ytest.size),
        'Prediction':Ytest.astype(int)})
    out.to_csv(filename,index=False)

def report(grid_scores, n_top=5):
    params = None
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Parameters with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
              score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

        if params == None:
            params = score.parameters

    return params


def Gridsearch_impl(X,Y,clf,param,cv):

    grid_search = GridSearchCV(clf,param,verbose=10,cv=cv,n_jobs=10)
    start = time()
    grid_search.fit(X,Y)
#    print(grid_search.grid_scores_)
    best = report(grid_search.grid_scores_)


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
    print("gradient boosting  classifier!")

    X,Y,Xtest = importdata()
    print(Y.shape)
    param_grid={
            "n_estimators":[10,100,200,2000,20000],
            "min_samples_split":[5,10,20,50]
            }

    gb=GradientBoostingClassifier()
    Gridsearch_impl(X,Y,gb,param_grid,5)

#    for i in range(10,11,5):
#        clf = DecisionTreeClassifier(min_samples_split=i)
#        rf = RandomForestClassifier(n_estimators = 100,random_state=0,min_samples_split=i)
#        ab = AdaBoostClassifier(rf,n_estimators = 10)
        #ab = GradientBoostingClassifier(n_estimators = 100)
#        score = cross_validation.cross_val_score(ab,X,Y,cv=3)
      #  print(score)
      #  print("average score %f"%np.mean(score))
      #  print("std %f"%np.std(score))
      #  ab.fit(X,Y)
   


    Ytest = gb.predict(Xtest)
    output(Ytest,'submit3.csv')


if __name__ == '__main__':
    main()
