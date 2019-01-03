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


### SVC area, which is used for the model1's classfication part
def svc_result(data_train, data_test, label_train, label_test):
    if data_train.shape[-1] == 0:
        return 0
    else:
        model = SVC()
        model.fit(data_train, label_train)
        preditcion = model.predict(data_test)
        fpr, tpr, _ = roc_curve(
            label_test.ravel(), preditcion.ravel(), pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc


### model1 area


## This is defination of laplace matrix
def laplace_matrix(n, data):
    cc_matrix = np.corrcoef(data.T)
    cc_matrix = np.abs(cc_matrix)
    adj_matrix = cc_matrix - np.eye(n)
    normlize_matrix = np.diag(
        np.dot(adj_matrix,
               np.repeat(1., n).reshape((n, 1)).ravel()))
    normlize_matrix = np.linalg.inv(np.sqrt(normlize_matrix))
    laplace = np.eye(n) - np.dot(
        np.dot(normlize_matrix, adj_matrix), normlize_matrix)
    return laplace


## This is the defination of data y, which represents the relationship of the label and gene expression of each sample
def initial_y(n, data, label):
    y = np.zeros(n).reshape(n, 1)
    for i in range(n):
        y[i] = np.corrcoef(data.T[i, :], label.ravel())[1, 0]
    y = np.abs(y)
    return y


## this is the initialzation of feature selection list, all elements are set to be small amount
def initial_f(n):
    f1 = np.repeat(1. / n, n).reshape(n, 1)
    f2 = f1.copy()
    fc = f1.copy()
    return f1, f2, fc


## In the approach process, this function is used to update the feature selection list fi. i=1,2
def model1_update_fi(n, laplace, fi, fc, y, alpha, gamma):
    Q = np.linalg.cholesky(alpha * laplace + (1 - alpha) * np.eye(n))
    Y = -(np.dot(fc.T, Q.T - (1 - alpha) * np.dot(y.T, np.linalg.inv(Q)))).T
    model = linear_model.Lasso(1 / 2 * gamma)
    model.fit(Q, Y)
    fi = model.coef_
    fi = fi.reshape((n, 1))
    return fi


## In the approach process, this function is used to update the common feature selection list fc.
def model1_update_fc(n, laplace1, laplace2, f1, y1, f2, y2, fc, alpha, gamma):
    Q = np.linalg.cholesky(alpha * laplace1 + 2 * (1 - alpha) * np.eye(n) +
                           alpha * laplace2)
    Y = -(np.dot((alpha * np.dot(f1.T, laplace1) + (1 - alpha) * (f1 - y1).T) +
                 (alpha * np.dot(f2.T, laplace2) +
                  (1 - alpha) * (f2 - y2).T), np.linalg.inv(Q))).T
    model = linear_model.Lasso(1 / 2 * gamma)
    model.fit(Q, Y)
    fc = model.coef_
    fc = fc.reshape((1, len(fc)))
    return fc


## Once the model1 function generate the feature selection list already, this function can generate the roc value
## based on the list to evaluate the model's performance
def model1_roc(n, f1, f2, data1, data2, label1_train, label1_test,
               label2_train, label2_test):
    f1[f1 > 0] = 1
    f1[f1 <= 0] = 0
    f2[f2 > 0] = 1
    f2[f2 <= 0] = 0
    left1_sel = np.diag(f1.ravel())
    left2_sel = np.diag(f2.ravel())
    data1 = np.dot(left1_sel, data1.T)
    data2 = np.dot(left2_sel, data2.T)
    ind = []
    for i in range(n):
        temp1 = data1[i, :]
        if len(temp1[temp1 == 0]) > 20:
            ind.append(i)
    data1 = np.delete(data1, ind, axis=0)
    ind = []
    for i in range(n):
        temp2 = data2[i, :]
        if len(temp2[temp2 == 0]) > 20:
            ind.append(i)
    data2 = np.delete(data2, ind, axis=0)
    data1 = data1.T
    data1_train = data1[:len(label1_train), :]
    data1_test = data1[len(label1_train):, :]
    data2 = data2.T
    data2_train = data2[:len(label2_train), :]
    data2_test = data2[len(label2_train):, :]
    roc1 = svc_result(data1_train, data1_test, label1_train, label1_test)
    roc2 = svc_result(data2_train, data2_test, label2_train, label2_test)
    return roc1, roc2


## this function trains model1 on the training data set, and output the roc values which is based on the model's
## performance on the validation data
def train_model1(n,data1_train, data1_validation, label1_train,label1_validation, \
                 data2_train, data2_validation, label2_train, label2_validation, alpha,gamma1,gamma2,gamma3):
    data1 = np.concatenate((data1_train, data1_validation), axis=0)
    data2 = np.concatenate((data2_train, data2_validation), axis=0)
    laplace1 = laplace_matrix(n, data1)
    laplace2 = laplace_matrix(n, data2)
    y1 = initial_y(n, data1_train, label1_train)
    y2 = initial_y(n, data2_train, label2_train)
    f1, f2, fc = initial_f(n)
    for interation in range(50):
        f1 = f1.reshape((n, 1))
        f2 = f2.reshape((n, 1))
        fc = fc.reshape((n, 1))
        f1_ori = f1.copy()
        f1 = model1_update_fi(n, laplace1, f1, fc, y1, alpha, gamma1)
        f2_ori = f2.copy()
        f2 = model1_update_fi(n, laplace2, f2, fc, y2, alpha, gamma2)
        fc_ori = fc.copy()
        fc = model1_update_fc(n, laplace1, laplace2, f1, y1, f2, y2, fc, alpha,
                              gamma3)
        if np.sum(np.abs(f1 - f1_ori)) > 100:
            break
        if np.sum(np.abs(f1 - f1_ori)) < 1e-5 or \
        np.sum(np.abs(f2 - f2_ori)) < 1e-5 or \
        np.sum(np.abs(fc - fc_ori)) < 1e-5:
            break
    roc1_train, roc2_train = model1_roc(n, f1, f2, data1, data2, label1_train,
                                        label1_validation, label2_train,
                                        label2_validation)
    return roc1_train, roc2_train


## this function trains the model1 on the same training data set and output its roc value
## to evaluate its performance on the test data sets
def test_model1(n,data1_train, data1_validation, data1_test, label1_train,label1_validation, label1_test,\
           data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test,alpha,gamma1,gamma2,gamma3):
    data1 = np.concatenate((data1_train, data1_validation), axis=0)
    data2 = np.concatenate((data2_train, data2_validation), axis=0)
    data1_train = np.concatenate((data1, data1_test), axis=0)
    data2_train = np.concatenate((data2, data2_test), axis=0)
    label1_train = np.concatenate((label1_train, label1_validation), axis=0)
    label2_train = np.concatenate((label2_train, label2_validation), axis=0)
    label1_temp = np.concatenate(
        (label1_train, np.repeat(0., len(label1_test)).reshape(
            (len(label1_test), 1))),
        axis=0)
    label2_temp = np.concatenate(
        (label2_train, np.repeat(0., len(label2_test)).reshape(
            (len(label2_test), 1))),
        axis=0)
    laplace1 = laplace_matrix(n, data1)
    laplace2 = laplace_matrix(n, data2)
    y1 = initial_y(n, data1_train, label1_temp)
    y2 = initial_y(n, data2_train, label2_temp)
    f1, f2, fc = initial_f(n)
    for interation in range(50):
        f1 = f1.reshape((n, 1))
        f2 = f2.reshape((n, 1))
        fc = fc.reshape((n, 1))
        f1_ori = f1.copy()
        f1 = model1_update_fi(n, laplace1, f1, fc, y1, alpha, gamma1)
        f2_ori = f2.copy()
        f2 = model1_update_fi(n, laplace2, f2, fc, y2, alpha, gamma2)
        fc_ori = fc.copy()
        fc = model1_update_fc(n, laplace1, laplace2, f1, y1, f2, y2, fc, alpha,
                              gamma3)
        if np.sum(np.abs(f1 - f1_ori)) > 100:
            break
        if np.sum(np.abs(f1 - f1_ori)) < 1e-5 or \
        np.sum(np.abs(f2 - f2_ori)) < 1e-5 or \
        np.sum(np.abs(fc - fc_ori)) < 1e-5:
            break
    roc1_test, roc2_test = model1_roc(n, f1, f2, data1_train, data2_train,
                                      label1_train, label1_test, label2_train,
                                      label2_test)
    return roc1_test, roc2_test


### model1 main area
## this function utilize the functions above to output all the roc values of model1 based on the training data sets,
## validation data sets, test data sets
def model1(n,data1_train, data1_validation, data1_test, label1_train,label1_validation, label1_test,\
           data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test,alpha,gamma1,gamma2,gamma3):
    roc1_train,roc2_train=train_model1(n,data1_train, data1_validation, label1_train,label1_validation,\
                                       data2_train, data2_validation, label2_train, label2_validation, alpha,gamma1,gamma2,gamma3)
    roc1_test, roc2_test=test_model1(n,data1_train, data1_validation, data1_test, label1_train,label1_validation, label1_test,\
                                     data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test,alpha,gamma1,gamma2,gamma3)

    return roc1_train, roc2_train, roc1_test, roc2_test


## This function runs model1 all over the possible parameters which are defined
def model1_roc_all(n,data1_train, data1_validation, data1_test, label1_train,label1_validation, label1_test,\
           data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test):
    alpha = 0.1
    gamma1_all = np.arange(5e-5, 16e-5, 5e-5)
    gamma2_all = np.arange(5e-5, 16e-5, 5e-5)
    gamma3_all = np.arange(5e-5, 16e-5, 5e-5)
    model1_roc_train = np.zeros((3, 3, 3))
    model2_roc_train = np.zeros((3, 3, 3))
    model1_roc_test = np.zeros((3, 3, 3))
    model2_roc_test = np.zeros((3, 3, 3))
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                roc1_train,roc2_train, roc1_test, roc2_test=model1(n,data1_train, data1_validation,\
                                                                   data1_test, label1_train,label1_validation,\
                                                                   label1_test,data2_train, data2_validation,\
                                                                   data2_test, label2_train, label2_validation,\
                                                                   label2_test,alpha,gamma1_all[iii],gamma2_all[jjj],gamma3_all[kkk])
                model1_roc_train[iii, jjj, kkk] += roc1_train
                model2_roc_train[iii, jjj, kkk] += roc2_train
                model1_roc_test[iii, jjj, kkk] += roc1_test
                model2_roc_test[iii, jjj, kkk] += roc2_test
    return model1_roc_train, model2_roc_train, model1_roc_test, model2_roc_test


### This function records all models' results
def all_models(n,data1_train, data1_validation, data1_test,\
               label1_train,label1_validation, label1_test,\
               data2_train, data2_validation,data2_test, label2_train, label2_validation, label2_test):

    # model1 part
    model1_1_roc_train, model1_2_roc_train, model1_1_roc_test, model1_2_roc_test=model1_roc_all(n,data1_train, data1_validation, data1_test,\
                                                                                        label1_train,label1_validation, label1_test,\
                                                                                        data2_train, data2_validation, data2_test,\
                                                                                        label2_train, label2_validation, label2_test)
    loc1 = np.argmax(model1_1_roc_train)
    model1_1_roc_train_max = np.max(model1_1_roc_train)
    final_model1_1_roc = model1_1_roc_test[loc1 // 9][loc1 % 9 // 3][loc1 % 9 %
                                                                     3]
    loc2 = np.argmax(model1_2_roc_train)
    model1_2_roc_train_max = np.max(model1_2_roc_train)
    final_model1_2_roc = model1_2_roc_test[loc2 // 9][loc2 % 9 // 3][loc2 % 9 %
                                                                     3]

    return model1_1_roc_train_max, final_model1_1_roc, model1_2_roc_train_max, final_model1_2_roc


### This function is the cross validation process, it runs the CV process 50 times, and output every time's result
### in the txt file
def whole_process(n, data_1_all, data_2_all, label_1_all, label_2_all,
                  validation_number, test_number, preprocess_scale):
    ### run 50 times, get the average
    final_model1_1_roc_all = 0.
    final_model1_2_roc_all = 0.

    model1_1_roc_train_max_all = 0.
    model1_2_roc_train_max_all = 0.

    for i in range(50):
        data1_train, data1_validation, data1_test, label1_train, label1_validation,\
                  label1_test, data2_train, data2_validation, data2_test, label2_train,\
                  label2_validation, label2_test=cross_validation(data_1_all,data_2_all,\
                  label_1_all,label_2_all,validation_number,test_number)
        model1_1_roc_train_max,final_model1_1_roc, model1_2_roc_train_max,\
        final_model1_2_roc,base_roc1=all_models(n,data1_train, data1_validation, data1_test,\
                                                                                label1_train,label1_validation, label1_test,\
                                                                                data2_train, data2_validation,data2_test,\
                                                                                label2_train, label2_validation, label2_test)

        model1_1_roc_train_max_all += model1_1_roc_train_max
        final_model1_1_roc_all += final_model1_1_roc
        model1_2_roc_train_max_all += model1_2_roc_train_max
        final_model1_2_roc_all += final_model1_2_roc
        print(i)
        temp_report = open('temp_report_model1.txt' % preprocess_scale, 'a')
        print(file=temp_report)
        print('iterations:', i + 1, file=temp_report)

        print(model1_1_roc_train_max_all / (i + 1), file=temp_report)
        print(final_model1_1_roc_all / (i + 1), file=temp_report)
        print(model1_2_roc_train_max_all / (i + 1), file=temp_report)
        print(final_model1_2_roc_all / (i + 1), file=temp_report)
        temp_report.close()
    final_report = open('final_report_model1.txt' % preprocess_scale, 'w')
    print(model1_1_roc_train_max_all / (i + 1), file=final_report)
    print(final_model1_1_roc_all / (i + 1), file=final_report)
    print(model1_2_roc_train_max_all / (i + 1), file=final_report)
    print(final_model1_2_roc_all / (i + 1), file=final_report)
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
