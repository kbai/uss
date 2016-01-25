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
    trainf = './training_data.txt'
    testf = './testing_data.txt'
    train_data = np.loadtxt(trainf,delimiter='|',skiprows = 1)
    test_data = np.loadtxt(testf,delimiter='|',skiprows = 1)
    X = train_data[:,1:-1]
    Y = train_data[:,-1]
    N,D = X.shape
    for ii in range(0,D):
        if(np.sum(X[:,ii]) == 0.0):
            print("%d, feature all 0!"%ii)
#    for ii in range(0,79):
#        for jj in range(0,ii):
#            if( np.alltrue(X[ii,:] == X[jj,:])):
           #     print("pair %d, %d, %d, %d"%(ii,jj,Y[ii],Y[jj]))
           #     print(X[ii,:])
           #     print(X[jj,:])
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
    for i in range(2,50):
        clf = DecisionTreeClassifier(min_samples_split=i)
        #rf = RandomForestClassifier(n_estimators = 300)
        #ab = AdaBoostClassifier(n_estimators = 100)
        ab = GradientBoostingClassifier(n_estimators = 100)
        score = cross_validation.cross_val_score(ab,X,Y,cv=5)
        print("average score %f"%np.mean(score))
        print("std %f"%np.std(score))
        clf.fit(X,Y)
        print(clf.score(X,Y))
    
    #output_trainE(X,Y,clf)
    #PCA_analysis(X,100)


#    nleaf = 100
#    dt = DecisionTreeClassifier(min_samples_split = nleaf)
#    clf = AdaBoostClassifier(dt,algorithm="SAMME",n_estimators=200,random_state=nleaf)
#    clf.fit(X,Y)
#    Ytest=clf.predict(Xtest)
#output(Ytest,'adaboost_005_many_{}.csv'.format(nleaf))
#    Yt=clf.predict(X)
#    print(np.sum((np.abs(Y-Yt))))



if __name__ == '__main__':
    main()
