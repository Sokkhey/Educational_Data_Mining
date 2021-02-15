# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:32:38 2020

@author: Sokkhey
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#-- Model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

data = pd.read_csv('ADS.csv', low_memory=False)

#---------------------------------------------------------------------- Data Split
X = data.drop('SCORE', axis=1)
X= pd.get_dummies(X)
y = data['SCORE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=21)


#-----------------------------
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def evaluation_baselinemodel(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    model_accuracy = model.score(X_test, y_test)
    print('The accuracy of the model: {:.4f}'.format(model_accuracy))
    model_rmse = rmse(y_pred, y_test)
    print("The RMSE of the model:{}".format(model_rmse))
#---
def evaluation_baselinemodel1(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print(confusion_matrix(y_train, y_pred))
    model_accuracy = model.score(X_train, y_train)
    print('The accuracy of the model: {:.4f}'.format(model_accuracy))
    model_rmse = rmse(y_pred, y_train)
    print("The RMSE of the model:{}".format(model_rmse))
#-NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN--- Baseline
nb = GaussianNB()
evaluation_baselinemodel(nb)
# -- CART

DT = DecisionTreeClassifier()
print("The evaluation metrics of the DT:")
evaluation_baselinemodel(DT)
# -- RF

rfc = RandomForestClassifier()
print("The evaluation metrics of the RF:")
evaluation_baselinemodel1(rfc)

#-- SVM

svmc = svm.SVC()   # C=10, gamma=0.01
print("The evaluation metrics of the SVM:")
evaluation_baselinemodel(svmc)

#-- 10 run times
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rf_acc_vec = []
rf_rmse_vec = []
for i in range(10):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_train)
    model_accuracy = nb.score(X_train, y_train)
    model_rmse = rmse(y_pred, y_train)
    rf_acc_vec.append(model_accuracy)
    rf_rmse_vec.append(model_rmse)
    #print(rf_acc_vec)
print('Model Average of model is:{} and RMSE is: {}'.format(np.mean(rf_acc_vec), np.mean(rf_rmse_vec)))

rf_acc_vec = []
i_max = 10
while model_accuracy < i_max:
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_train)
    model_accuracy = nb.score(X_train, y_train)
    rf_acc_vec.append(model_accuracy)
    print(rf_acc_vec)

#-- NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN-- 10-CV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

data = pd.read_csv('ADS.csv', low_memory=False)
X = data.drop('SCORE', axis=1)
X= pd.get_dummies(X)
y = data['SCORE']

def evaluation_metrics(model):
    acc = cross_val_score(model, X, y, scoring='accuracy', cv=10, n_jobs=-1)
    scores = cross_validate(model, X, y, cv=10, scoring=('accuracy', 'neg_root_mean_squared_error'), return_train_score=True)
    rmse = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
    print('ACC of testing: %.4f' % np.mean(acc))  
    print('Recall: %.4f' % np.mean(rmse))
    print(scores)
    #print(acc1['train_neg_root_mean_squared_error'])
    
#---------------------------- RF
rfc = RandomForestClassifier()
model_scores = cross_validate(rfc, X, y, cv=10, scoring=('accuracy', 'neg_root_mean_squared_error'), return_train_score=True)
print('The testing accuracy is: {:.4f}'.format(np.mean(model_scores['test_accuracy'])))
print('The testing RMSE is: {:.4f}'.format(np.mean(model_scores['test_neg_root_mean_squared_error'])))
print('The training accuracy is: {:.4f}'.format(np.mean(model_scores['train_accuracy'])))
print('The training RMSE is: {:.4f}'.format(np.mean(model_scores['train_neg_root_mean_squared_error'])))

#---------------------------- NB
nb = GaussianNB()
model_scores = cross_validate(nb, X, y, cv=10, scoring=('accuracy', 'neg_root_mean_squared_error'), return_train_score=True)
print('The testing accuracy is: {:.4f}'.format(np.mean(model_scores['test_accuracy'])))
print('The testing RMSE is: {:.4f}'.format(np.mean(model_scores['test_neg_root_mean_squared_error'])))
print('The training accuracy is: {:.4f}'.format(np.mean(model_scores['train_accuracy'])))
print('The training RMSE is: {:.4f}'.format(np.mean(model_scores['train_neg_root_mean_squared_error'])))

#---------------------------- SVM
svmc = svm.SVC(C=100, gamma=1)  # C=10, gamma=0.01
model_scores = cross_validate(svmc, X, y, cv=10, scoring=('accuracy', 'neg_root_mean_squared_error'), return_train_score=True)
print('The testing accuracy is: {:.4f}'.format(np.mean(model_scores['test_accuracy'])))
print('The testing RMSE is: {:.4f}'.format(np.mean(model_scores['test_neg_root_mean_squared_error'])))
print('The training accuracy is: {:.4f}'.format(np.mean(model_scores['train_accuracy'])))
print('The training RMSE is: {:.4f}'.format(np.mean(model_scores['train_neg_root_mean_squared_error'])))




#-- NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN-- PCA

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn import preprocessing
from sklearn.decomposition import PCA

#---------------------------------------------------------------------- SVM + PCA

X = data.drop('SCORE', axis=1)
X= pd.get_dummies(X)
y = data['SCORE']
X = pd.DataFrame(PCA(n_components=25).fit_transform(X))
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=21)

#-------------------- svm test
svm_acc_vec = []
svm_rmse_vec = []
for i in range(10):
    svmc = svm.SVC(C=100, gamma=1)  
    svmc.fit(X_train, y_train)
    y_pred = svmc.predict(X_test)
    model_accuracy = svmc.score(X_test, y_test)
    model_rmse = rmse(y_pred, y_test)
    svm_acc_vec.append(model_accuracy)
    svm_rmse_vec.append(model_rmse)
    #print(rf_acc_vec)
print('Model Average of model is:{:.4f} and RMSE is: {:.4f}'.format(np.mean(svm_acc_vec), np.mean(svm_rmse_vec)))
lst1 = [np.mean(svm_acc_vec)*100, np.std(svm_acc_vec)*100, np.mean(svm_rmse_vec), np.std(svm_rmse_vec)]

#-------------------- svm train
svm_acc_vec = []
svm_rmse_vec = []
for i in range(10):
    svmc = svm.SVC(C=100, gamma=1)  
    svmc.fit(X_train, y_train)
    y_pred = svmc.predict(X_train)
    model_accuracy = svmc.score(X_train, y_train)
    model_rmse = rmse(y_pred, y_train)
    svm_acc_vec.append(model_accuracy)
    svm_rmse_vec.append(model_rmse)
    #print(rf_acc_vec)
print('Model Average of model is:{} and RMSE is: {}'.format(np.mean(svm_acc_vec), np.mean(svm_rmse_vec)))
lst2 = [np.mean(svm_acc_vec)*100, np.std(svm_acc_vec)*100, np.mean(svm_rmse_vec), np.std(svm_rmse_vec)]
df = pd.DataFrame(list(zip(lst1, lst2)), 
               columns =['Test', 'Train']) 
print(df)
print(lst1)
print(lst2)
#---------------------------- SVM + 10-CV + PCA

X = data.drop('SCORE', axis=1)
X= pd.get_dummies(X)
y = data['SCORE']
X = pd.DataFrame(PCA(n_components=25).fit_transform(X))
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
svmc = svm.SVC(C=100, gamma=1)  # C=10, gamma=0.01
model_scores = cross_validate(svmc, X, y, cv=10, scoring=('accuracy', 'neg_root_mean_squared_error'), return_train_score=True)
print('The testing accuracy is: {:.4f}'.format(np.mean(model_scores['test_accuracy'])))
print('The testing RMSE is: {:.4f}'.format(np.mean(model_scores['test_neg_root_mean_squared_error'])))
print('The training accuracy is: {:.4f}'.format(np.mean(model_scores['train_accuracy'])))
print('The training RMSE is: {:.4f}'.format(np.mean(model_scores['train_neg_root_mean_squared_error'])))