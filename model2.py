#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
import pickle
np.random.seed(20)
np.set_printoptions(threshold=np.inf)


### This function reads data sets from pkl files directly, after that, data cleaning process will be excuated too
def read_data(names):
    data_1 = pickle.load(open(names[0], 'rb'))
    label_1 = pickle.load(open(names[1], 'rb'))
    data_2 = pickle.load(open(names[2], 'rb'))
    label_2 = pickle.load(open(names[3], 'rb'))
    n = data_1.shape[1]
    return n, data_1, data_2, label_1, label_2


### output the data matrices, n is the number of features


### The defination of the cross valation nethod
def cross_validation(data_1_all, data_2_all, label_1_all, label_2_all,
                     validation_number, test_number):
    ### get the test part
    data_1, data1_test, label_1, label1_test = train_test_split(
        data_1_all, label_1_all, test_size=test_number)
    data_2, data2_test, label_2, label2_test = train_test_split(
        data_2_all, label_2_all, test_size=test_number)
    ### get the train and validation parts
    data1_train, data1_validation, label1_train, label1_validation = train_test_split(
        data_1, label_1, test_size=validation_number)
    data2_train, data2_validation, label2_train, label2_validation = train_test_split(
        data_2, label_2, test_size=validation_number)
    return data1_train, data1_validation, data1_test, label1_train,\
           label1_validation, label1_test, data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test


### model2 area
## this function defines the adjance matrix of model2 which is different from the model1's
def adjance_matrix(data):
    v = data.shape[0]
    u = data.shape[1]
    left_matrix = np.eye(v, v)
    right_matrix = np.eye(u, u)
    left_matrix = np.diag(
        np.dot(data,
               np.repeat(1., u).reshape((u, 1)).ravel()))
    right_matrix = np.diag(
        np.dot(np.repeat(1., v).reshape((1, v)).ravel(), data))
    left_matrix = np.linalg.inv(np.sqrt(left_matrix))
    right_matrix = np.linalg.inv(np.sqrt(right_matrix))
    adjance = np.dot(np.dot(left_matrix, data), right_matrix)
    return adjance.T


## This function initialized all the feature selection lists and lables
def initial_fv_fu(data1, data2, label1_train, label2_train):
    n = data1.shape[1]
    m1 = data1.shape[0]
    m2 = data2.shape[0]
    fv1 = np.repeat(1. / n, n).reshape((n, 1))
    fv2 = fv1.copy()
    fvc = fv1.copy()
    fu1 = np.concatenate(
        (label1_train, np.repeat(0., m1 - label1_train.shape[0]).reshape(
            (m1 - label1_train.shape[0], 1))),
        axis=0)
    fu2 = np.concatenate(
        (label2_train, np.repeat(0., m2 - label2_train.shape[0]).reshape(
            (m2 - label2_train.shape[0], 1))),
        axis=0)
    return fv1, fv2, fvc, fu1, fu2


## In the approach process, this function is used to update the feature selection list fv. v=1,2
def model2_update_fvi(n, adjance, fui, yvi, fvc, alpha, gamma):
    Y = (np.dot(adjance, fui) + alpha * yvi) / (1 + alpha) - fvc
    model = linear_model.Lasso(1 / 2 * gamma)
    Q = np.eye(n)
    model.fit(Q, Y)
    fvi = model.coef_
    fvi = fvi.reshape((n, 1))
    return fvi


## In the approach process, this function is used to update the common feature selection list fc.
def model2_update_fvc(n, adjance1, adjance2, fu1, fu2, fv1, fv2, yv1, yv2,
                      alpha, gamma):
    Y = 1 / 2 * ((np.dot(adjance1, fu1) + alpha * yv1) / (1 + alpha) - fv1 +
                 (np.dot(adjance2, fu2) + alpha * yv2) / (1 + alpha) - fv2)
    model = linear_model.Lasso(1 / 2 * gamma)
    Q = np.eye(n)
    model.fit(Q, Y)
    fvc = model.coef_
    fvc = fvc.reshape((n, 1))
    return fvc


## In the approach process, this function is used to update the label's prediction.
def model2_update_fui(n, adjance, fvi, fvc, alpha, yui):
    fui = (np.dot(adjance.T, fvi) + np.dot(adjance.T, fvc) + alpha * yui) / (
        1 + alpha)
    return fui


## this function trains model2 with given data sets and defined parameters, and return
## the roc values of model on validation data sets
def train_model(n,data1_train, data1_validation, label1_train,label1_validation, data2_train,\
                data2_validation, label2_train, label2_validation,alpha,gamma1,gamma2,gamma3):
    data1 = np.concatenate((data1_train, data1_validation), axis=0)
    data2 = np.concatenate((data2_train, data2_validation), axis=0)
    fv1, fv2, fvc, fu1, fu2 = initial_fv_fu(data1, data2, label1_train,
                                           label2_train)
    label1_temp = np.concatenate(
        (label1_train, np.repeat(0., len(label1_validation)).reshape(
            (len(label1_validation)), 1)),
        axis=0)
    label2_temp = np.concatenate(
        (label2_train, np.repeat(0., len(label2_validation)).reshape(
            (len(label2_validation)), 1)),
        axis=0)
    yv1 = initial_y(n, data1, label1_temp)
    yv2 = initial_y(n, data2, label2_temp)
    adjance1 = adjance_matrix(data1)
    adjance2 = adjance_matrix(data2)
    yu1 = fu1.copy()
    yu2 = fu2.copy()
    for i in range(50):
        fv1 = fv1.reshape((n, 1))
        fv2 = fv2.reshape((n, 1))
        fvc = fvc.reshape((n, 1))
        fv1_ori = fv1.copy()
        fv1 = model2_update_fvi(n, adjance1, fu1, yv1, fvc, alpha, gamma1)
        fv2_ori = fv2.copy()
        fv2 = model2_update_fvi(n, adjance2, fu2, yv2, fvc, alpha, gamma2)
        fvc_ori = fvc.copy()
        fvc = model2_update_fvc(n, adjance1, adjance2, fu1, fu2, fv1, fv2, yv1,
                                yv2, alpha, gamma3)
        fu1 = model2_update_fui(n, adjance1, fv1, fvc, alpha, yu1)
        fu2 = model2_update_fui(n, adjance2, fv2, fvc, alpha, yu2)
        if np.sum(np.abs(fv1 - fv1_ori)) > 100:
            break
        if np.sum(np.abs(fv1 - fv1_ori)) < 1e-5 or \
        np.sum(np.abs(fv2 - fv2_ori)) < 1e-5 or \
        np.sum(np.abs(fvc - fvc_ori)) < 1e-5:
            break
    fu1_test = fu1[len(fu1) - len(label1_validation):]
    fpr, tpr, _ = roc_curve(
        label1_validation.ravel(), fu1_test.ravel(), pos_label=-1)
    roc_auc1 = auc(fpr, tpr)
    fu2_test = fu2[len(fu2) - len(label2_validation):]
    fpr, tpr, _ = roc_curve(
        label2_validation.ravel(), fu2_test.ravel(), pos_label=-1)
    roc_auc2 = auc(fpr, tpr)
    return roc_auc1, roc_auc2


## This function gets the model2's performance on test data sets with give parameters
def test_model(n,data1_train, data1_validation, data1_test, label1_train,label1_validation, label1_test,\
           data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test,alpha,gamma1,gamma2,gamma3):
    data1 = np.concatenate((data1_train, data1_test), axis=0)
    data2 = np.concatenate((data2_train, data2_test), axis=0)
    label1_temp = np.concatenate(
        (label1_train, np.repeat(0., len(label1_test)).reshape(
            (len(label1_test)), 1)),
        axis=0)
    label2_temp = np.concatenate(
        (label2_train, np.repeat(0., len(label2_test)).reshape(
            (len(label2_test)), 1)),
        axis=0)
    fv1, fv2, fvc, fu1, fu2 = initial_fv_fu(data1, data2, label1_train,
                                           label2_train)
    yv1 = initial_y(n, data1, label1_temp)
    yv2 = initial_y(n, data2, label2_temp)
    adjance1 = adjance_matrix(data1)
    adjance2 = adjance_matrix(data2)
    yu1 = fu1.copy()
    yu2 = fu2.copy()
    for i in range(50):
        fv1 = fv1.reshape((n, 1))
        fv2 = fv2.reshape((n, 1))
        fvc = fvc.reshape((n, 1))
        fv1_ori = fv1.copy()
        fv1 = model2_update_fvi(n, adjance1, fu1, yv1, fvc, alpha, gamma1)
        fv2_ori = fv2.copy()
        fv2 = model2_update_fvi(n, adjance2, fu2, yv2, fvc, alpha, gamma2)
        fvc_ori = fvc.copy()
        fvc = model2_update_fvc(n, adjance1, adjance2, fu1, fu2, fv1, fv2, yv1,
                                yv2, alpha, gamma3)
        fu1 = model2_update_fui(n, adjance1, fv1, fvc, alpha, yu1)
        fu2 = model2_update_fui(n, adjance2, fv2, fvc, alpha, yu2)
        if np.sum(np.abs(fv1 - fv1_ori)) > 100:
            break
        if np.sum(np.abs(fv1 - fv1_ori)) < 1e-5 or \
        np.sum(np.abs(fv2 - fv2_ori)) < 1e-5 or \
        np.sum(np.abs(fvc - fvc_ori)) < 1e-5:
            break
    fu1_test = fu1[len(fu1) - len(label1_test):]
    fpr, tpr, _ = roc_curve(
        label1_test.ravel(), fu1_test.ravel(), pos_label=-1)
    roc_auc1 = auc(fpr, tpr)
    fu2_test = fu2[len(fu2) - len(label2_test):]
    fpr, tpr, _ = roc_curve(
        label2_test.ravel(), fu2_test.ravel(), pos_label=-1)
    roc_auc2 = auc(fpr, tpr)
    return roc_auc1, roc_auc2


## this function recorders all the roc values of model2 on validation data and test data
def model2(n,data1_train, data1_validation, data1_test, label1_train,label1_validation, label1_test,\
           data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test,alpha,gamma1,gamma2,gamma3):
    train_roc_auc1, train_roc_auc2=train_model(n,data1_train, data1_validation, label1_train,label1_validation, data2_train,\
                                              data2_validation, label2_train, label2_validation,alpha,gamma1,gamma2,gamma3)
    test_roc_auc1, test_roc_auc2=test_model(n,data1_train, data1_validation, data1_test, label1_train,label1_validation, label1_test,\
                                            data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test,alpha,gamma1,gamma2,gamma3)
    return train_roc_auc1, train_roc_auc2, test_roc_auc1, test_roc_auc2


## This function records best roc values with given all possible parameters
def model2_roc_all(n,data1_train, data1_validation, data1_test, label1_train,label1_validation, label1_test,\
                   data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test):
    alpha = 0.01
    gamma1_all = np.arange(5e-8, 16e-8, 2e-8)
    gamma2_all = np.arange(5e-8, 16e-8, 2e-8)
    gamma3_all = np.arange(5e-8, 16e-8, 2e-8)
    model1_roc_train = np.zeros((6, 6, 6))
    model2_roc_train = np.zeros((6, 6, 6))
    model1_roc_test = np.zeros((6, 6, 6))
    model2_roc_test = np.zeros((6, 6, 6))
    for iii in range(6):
        for jjj in range(6):
            for kkk in range(6):
                train_roc_auc1, train_roc_auc2, test_roc_auc1, test_roc_auc2=model2(n,data1_train, data1_validation,\
                                                                                    data1_test, label1_train,label1_validation,\
                                                                                    label1_test,data2_train, data2_validation,\
                                                                                    data2_test, label2_train, label2_validation,\
                                                                                    label2_test,alpha,gamma1_all[iii],gamma2_all[jjj],gamma3_all[kkk])
                model1_roc_train[iii, jjj, kkk] += train_roc_auc1
                model2_roc_train[iii, jjj, kkk] += train_roc_auc2
                model1_roc_test[iii, jjj, kkk] += test_roc_auc1
                model2_roc_test[iii, jjj, kkk] += test_roc_auc2
    return model1_roc_train, model2_roc_train, model1_roc_test, model2_roc_test


### This function records all models' results
def all_models(n,data1_train, data1_validation, data1_test,\
               label1_train,label1_validation, label1_test,\
               data2_train, data2_validation,data2_test, label2_train, label2_validation, label2_test):

    # model2 part
    model2_1_roc_train, model2_2_roc_train, model2_1_roc_test, model2_2_roc_test=model2_roc_all(n,data1_train, data1_validation, data1_test,\
                                                                                                 label1_train,label1_validation, label1_test,data2_train,\
                                                                                                 data2_validation, data2_test, label2_train, label2_validation,\
                                                                                                 label2_test)
    loc1 = np.argmin(model2_1_roc_train)
    model2_1_roc_train_max = np.max(model2_1_roc_train)
    final_model2_1_roc = model2_1_roc_test[loc1 // 36][loc1 % 36 // 6][loc1 %
                                                                       36 % 6]
    loc2 = np.argmin(model2_2_roc_train)
    model2_2_roc_train_max = np.max(model2_2_roc_train)
    final_model2_2_roc = model2_2_roc_test[loc2 // 36][loc2 % 36 // 6][loc2 %
                                                                       36 % 6]

    return model2_1_roc_train_max, final_model2_1_roc, model2_2_roc_train_max, final_model2_2_roc


### This function is the cross validation process, it runs the CV process 50 times, and output every time's result
### in the txt file
def whole_process(n, data_1_all, data_2_all, label_1_all, label_2_all,
                  validation_number, test_number, preprocess_scale):
    ### run 50 times, get the average

    final_model2_1_roc_all = 0.
    final_model2_2_roc_all = 0.

    model2_1_roc_train_max_all = 0.
    model2_2_roc_train_max_all = 0.

    for i in range(50):
        data1_train, data1_validation, data1_test, label1_train, label1_validation,\
                  label1_test, data2_train, data2_validation, data2_test, label2_train,\
                  label2_validation, label2_test=cross_validation(data_1_all,data_2_all,\
                  label_1_all,label_2_all,validation_number,test_number)
        model2_1_roc_train_max,final_model2_1_roc,model2_2_roc_train_max, final_model2_2_roc=all_models(n,data1_train, data1_validation, data1_test,\
                                                                                label1_train,label1_validation, label1_test,\
                                                                                data2_train, data2_validation,data2_test,\
                                                                                label2_train, label2_validation, label2_test)

        model2_1_roc_train_max_all += model2_1_roc_train_max
        final_model2_1_roc_all += final_model2_1_roc
        model2_2_roc_train_max_all += model2_2_roc_train_max
        final_model2_2_roc_all += final_model2_2_roc

        print(i)
        temp_report = open('trian_report_model2.txt', 'a')
        print(file=temp_report)
        print('iterations:', i + 1, file=temp_report)
        print(model2_1_roc_train_max, file=temp_report)
        print(final_model2_1_roc, file=temp_report)
        print(model2_2_roc_train_max, file=temp_report)
        print(final_model2_2_roc, file=temp_report)

        temp_report.close()
    final_report = open('final_report_model2.txt', 'w')
    print(model2_1_roc_train_max_all / (i + 1), file=final_report)
    print(final_model2_1_roc_all / (i + 1), file=final_report)
    print(model2_2_roc_train_max_all / (i + 1), file=final_report)
    print(final_model2_2_roc_all / (i + 1), file=final_report)
    final_report.close()


###  main area

if __name__ == "__main__":
    ### This is the parameter area
    names = [
        'sample_breast_expression.pk', 'sample_breast_label.pk',
        'sample_ov_expression.pk', 'sample_ov_label.pk'
    ]
    test_number = 20
    validation_number = 20
    n, data_1_all, data_2_all, label_1_all, label_2_all = read_data(names)
    whole_process(n, data_1_all, data_2_all, label_1_all, label_2_all,
                  validation_number, test_number, preprocess_scale)
