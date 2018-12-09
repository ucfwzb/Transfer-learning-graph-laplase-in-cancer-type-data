#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
#from keras.layers import Dense
#import keras
import pickle
np.random.seed(20)
np.set_printoptions(threshold=np.inf)

### common area
def read_data(preprocess_scale):  
    D1=pickle.load(open(os.getcwd()+'/brca_total.pkl','rb'))        
    label_1=D1[0].values
    label_1=np.array(label_1)
    D2=pickle.load(open(os.getcwd()+'/ov_total.pkl','rb'))
    label_2=D2[0].values
    label_2=np.array(label_2)
    label_1[label_1<0.5]=-1
    label_1[label_1>-0.5]=1
    label_2[label_2<0.5]=-1
    label_2[label_2>-0.5]=1
    label_1=label_1.reshape((len(label_1),1))
    label_2=label_2.reshape((len(label_2),1))
    if preprocess_scale:
        data_1=pickle.load(open(os.getcwd()+'/data1_dl.pkl','rb'))
        data_2=pickle.load(open(os.getcwd()+'/data2_dl.pkl','rb'))
        data_1=data_1.T
        data_2=data_2.T
        n=data_1.shape[1]
        del D1, D2
    else:
        D1=D1.drop(0,1)
        D1=D1.T
        D1=D1.values
        D2=D2.drop(0,1)
        D2=D2.T
        D2=D2.values
        data_1 =D1
        data_2 =D2
### Here the matrix is feature x sample
        
### delete the features which have more than 30 0's
    length=data_1.shape[0]
    ind = []
    for i in range(length):
        temp1=data_1[i, :]
        temp2=data_2[i, :]
        if len(temp1[temp1==0])>20:
            ind.append(i)
        if len(temp2[temp2==0])>20:
            ind.append(i)
    ind =list(set(ind))
    data_1 = np.delete(data_1, ind, axis=0)
    data_2 = np.delete(data_2, ind, axis=0)
### delete the features which have low mean and variance 
    line_1_var=[]
    line_2_var=[]
    line_1_mean=[]
    line_2_mean=[]
    for i in range(data_1.shape[0]):
        line_1_var.append([i,np.var(data_1[i])])
    for i in range(data_2.shape[0]):
        line_2_var.append([i,np.var(data_2[i])])
    for i in range(data_1.shape[0]):
        line_1_mean.append([i,np.mean(data_1[i])])
    for i in range(data_1.shape[0]):
        line_2_mean.append([i,np.mean(data_2[i])])
    line_1_mean=sorted(line_1_mean,key=lambda x:x[1])
    line_2_mean=sorted(line_2_mean,key=lambda x:x[1])   
    line_1_var=sorted(line_1_var,key=lambda x:x[1])
    line_2_var=sorted(line_2_var,key=lambda x:x[1])
    index_1_var=[x[0] for x in line_1_var][:int(round(len(line_1_var)*0.5,0))]
    index_2_var=[x[0] for x in line_2_var][:int(round(len(line_2_var)*0.5,0))]
    index_1_mean=[x[0] for x in line_1_mean][:int(round(len(line_1_mean)*0.5,0))]
    index_2_mean=[x[0] for x in line_2_mean][:int(round(len(line_2_mean)*0.5,0))]
    index_1=set(index_1_mean).union(set(index_1_var))
    index_2=set(index_2_mean).union(set(index_2_var))
    index_all=index_1.intersection(index_2)
    index_all=list(set(index_all))
    data_1 = np.delete(data_1, index_all, axis=0)
    data_2 = np.delete(data_2, index_all, axis=0)
    data_1=data_1.T
    data_2=data_2.T
    n=data_1.shape[1]
    return n,data_1,data_2,label_1,label_2
### output matrix with shape sample x features, n is the number of features


### cross nethod
def cross_valudation(data_1_all,data_2_all,label_1_all,label_2_all,valudation_number,test_number):
### get the test part
    data_1, data1_test, label_1, label1_test=train_test_split(data_1_all, label_1_all,test_size=test_number) 
    data_2, data2_test, label_2, label2_test=train_test_split(data_2_all, label_2_all,test_size=test_number) 
### get the train and valudation parts
    data1_train, data1_valudation, label1_train, label1_valudation=train_test_split(data_1, label_1,test_size=valudation_number) 
    data2_train, data2_valudation, label2_train, label2_valudation=train_test_split(data_2, label_2,test_size=valudation_number) 
    return data1_train, data1_valudation, data1_test, label1_train,\
           label1_valudation, label1_test, data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test





### SVC area
    
def svc_result(data_train,data_test,label_train,label_test):
    if data_train.shape[-1]==0:
        return 0
    else:     
        model=SVC()
        model.fit(data_train,label_train)    
        preditcion=model.predict(data_test)
        fpr, tpr, _ = roc_curve(label_test.ravel(), preditcion.ravel(), pos_label=1)
        roc_auc = auc(fpr, tpr)
        return roc_auc
    

### model1 area
## get laplace matrix
def laplace_matrix(n, data):
    cc_matrix=np.corrcoef(data.T)
    cc_matrix=np.abs(cc_matrix)
    adj_matrix=cc_matrix-np.eye(n)
    normlize_matrix=np.diag(np.dot(adj_matrix,np.repeat(1.,n).reshape((n,1)).ravel()))  
    normlize_matrix=np.linalg.inv(np.sqrt(normlize_matrix))    
    laplace=np.eye(n)-np.dot(np.dot(normlize_matrix,adj_matrix),normlize_matrix)
    return laplace

## get y
def intial_y(n,data,label):
    y=np.zeros(n).reshape(n,1)
    for i in range(n):
        y[i]=np.corrcoef(data.T[i,:],label.ravel())[1, 0]
    y=np.abs(y)
    return y

def intial_f(n):
    f1=np.repeat(1./n,n).reshape(n,1)
    f2=f1.copy()
    fc=f1.copy()
    return f1,f2,fc

def model1_update_fi(n,laplace,fi,fc,y,alpha,gamma):
    Q=np.linalg.cholesky(alpha*laplace+(1-alpha)*np.eye(n))
    Y=-(np.dot(fc.T,Q.T-(1-alpha)*np.dot(y.T,np.linalg.inv(Q)))).T
    model = linear_model.Lasso(1 / 2 * gamma)
    model.fit(Q, Y)
    fi = model.coef_
    fi=fi.reshape((n,1))
    return fi


def model1_update_fc(n,laplace1,laplace2,f1,y1,f2,y2,fc,alpha,gamma):
    Q=np.linalg.cholesky(alpha*laplace1+2*(1-alpha)*np.eye(n)+alpha*laplace2)
    Y=-(np.dot((alpha*np.dot(f1.T,laplace1)+(1-alpha)*(f1-y1).T)+(alpha*np.dot(f2.T,laplace2)+(1-alpha)*(f2-y2).T),np.linalg.inv(Q))).T
    model= linear_model.Lasso(1 / 2 * gamma)
    model.fit(Q, Y)
    fc = model.coef_
    fc = fc.reshape((1, len(fc)))
    return fc

def model1_roc(n,f1,f2,data1,data2,label1_train,label1_test,label2_train,label2_test):
    f1[f1>0]=1
    f1[f1<=0]=0
    f2[f2>0]=1
    f2[f2<=0]=0
    left1_sel=np.diag(f1.ravel())
    left2_sel=np.diag(f2.ravel())
    data1=np.dot(left1_sel,data1.T)
    data2=np.dot(left2_sel,data2.T)
    ind = []
    for i in range(n):
        temp1=data1[i, :]
        if len(temp1[temp1==0])>20:
            ind.append(i)
    data1 = np.delete(data1, ind, axis=0) 
    ind = []
    for i in range(n):
        temp2=data2[i, :]
        if len(temp2[temp2==0])>20:
            ind.append(i)
    data2 = np.delete(data2, ind, axis=0) 
    data1=data1.T
    data1_train=data1[:len(label1_train),:]
    data1_test=data1[len(label1_train):,:]
    data2=data2.T
    data2_train=data2[:len(label2_train),:]
    data2_test=data2[len(label2_train):,:]
    roc1=svc_result(data1_train,data1_test,label1_train,label1_test)
    roc2=svc_result(data2_train,data2_test,label2_train,label2_test)
    return roc1, roc2
        

def train_model1(n,data1_train, data1_valudation, label1_train,label1_valudation, \
                 data2_train, data2_valudation, label2_train, label2_valudation, alpha,gamma1,gamma2,gamma3):
    data1=np.concatenate((data1_train,data1_valudation),axis=0)
    data2=np.concatenate((data2_train,data2_valudation),axis=0)
    laplace1=laplace_matrix(n, data1)
    laplace2=laplace_matrix(n, data2)
    y1=intial_y(n,data1_train,label1_train)
    y2=intial_y(n,data2_train,label2_train)
    f1,f2,fc=intial_f(n)
    for interation in range(50):
        f1=f1.reshape((n,1))
        f2=f2.reshape((n,1))
        fc=fc.reshape((n,1))
        f1_ori=f1.copy()
        f1=model1_update_fi(n,laplace1,f1,fc,y1,alpha,gamma1)
        f2_ori=f2.copy()
        f2=model1_update_fi(n,laplace2,f2,fc,y2,alpha,gamma2)
        fc_ori=fc.copy()
        fc=model1_update_fc(n,laplace1,laplace2,f1,y1,f2,y2,fc,alpha,gamma3)
        if np.sum(np.abs(f1-f1_ori))>100:
            break
        if np.sum(np.abs(f1 - f1_ori)) < 1e-5 or \
        np.sum(np.abs(f2 - f2_ori)) < 1e-5 or \
        np.sum(np.abs(fc - fc_ori)) < 1e-5:
            break
    roc1_train, roc2_train= model1_roc(n,f1,f2,data1,data2,label1_train,label1_valudation,label2_train,label2_valudation)
    return roc1_train,roc2_train
    
def test_model1(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
           data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test,alpha,gamma1,gamma2,gamma3):
    data1=np.concatenate((data1_train,data1_valudation),axis=0)
    data2=np.concatenate((data2_train,data2_valudation),axis=0)
    data1_train=np.concatenate((data1,data1_test),axis=0)
    data2_train=np.concatenate((data2,data2_test),axis=0)
    label1_train=np.concatenate((label1_train,label1_valudation),axis=0)
    label2_train=np.concatenate((label2_train,label2_valudation),axis=0)
    label1_temp=np.concatenate((label1_train,np.repeat(0.,len(label1_test)).reshape((len(label1_test),1))),axis=0)
    label2_temp=np.concatenate((label2_train,np.repeat(0.,len(label2_test)).reshape((len(label2_test),1))),axis=0)
    laplace1=laplace_matrix(n, data1)
    laplace2=laplace_matrix(n, data2)
    y1=intial_y(n,data1_train,label1_temp)
    y2=intial_y(n,data2_train,label2_temp)
    f1,f2,fc=intial_f(n)
    for interation in range(50):
        f1=f1.reshape((n,1))
        f2=f2.reshape((n,1))
        fc=fc.reshape((n,1))
        f1_ori=f1.copy()
        f1=model1_update_fi(n,laplace1,f1,fc,y1,alpha,gamma1)
        f2_ori=f2.copy()
        f2=model1_update_fi(n,laplace2,f2,fc,y2,alpha,gamma2)
        fc_ori=fc.copy()
        fc=model1_update_fc(n,laplace1,laplace2,f1,y1,f2,y2,fc,alpha,gamma3)
        if np.sum(np.abs(f1-f1_ori))>100:
            break      
        if np.sum(np.abs(f1 - f1_ori)) < 1e-5 or \
        np.sum(np.abs(f2 - f2_ori)) < 1e-5 or \
        np.sum(np.abs(fc - fc_ori)) < 1e-5:
            break
    roc1_test, roc2_test= model1_roc(n,f1,f2,data1_train,data2_train,label1_train,label1_test,label2_train,label2_test)
    return roc1_test, roc2_test

 
### model1 main area   
def model1(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
           data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test,alpha,gamma1,gamma2,gamma3):
    roc1_train,roc2_train=train_model1(n,data1_train, data1_valudation, label1_train,label1_valudation,\
                                       data2_train, data2_valudation, label2_train, label2_valudation, alpha,gamma1,gamma2,gamma3)
    roc1_test, roc2_test=test_model1(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
                                     data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test,alpha,gamma1,gamma2,gamma3)
    
    return roc1_train,roc2_train, roc1_test, roc2_test

def model1_roc_all(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
           data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test):
    alpha = 0.1
    gamma1_all=np.arange(5e-5,16e-5,5e-5)
    gamma2_all=np.arange(5e-5,16e-5,5e-5)
    gamma3_all=np.arange(5e-5,16e-5,5e-5)
    model1_roc_train=np.zeros((3,3,3))
    model2_roc_train=np.zeros((3,3,3))
    model1_roc_test=np.zeros((3,3,3))
    model2_roc_test=np.zeros((3,3,3))
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                roc1_train,roc2_train, roc1_test, roc2_test=model1(n,data1_train, data1_valudation,\
                                                                   data1_test, label1_train,label1_valudation,\
                                                                   label1_test,data2_train, data2_valudation,\
                                                                   data2_test, label2_train, label2_valudation,\
                                                                   label2_test,alpha,gamma1_all[iii],gamma2_all[jjj],gamma3_all[kkk])
                model1_roc_train[iii,jjj,kkk]+=roc1_train
                model2_roc_train[iii,jjj,kkk]+=roc2_train
                model1_roc_test[iii,jjj,kkk]+=roc1_test
                model2_roc_test[iii,jjj,kkk]+=roc2_test
    return model1_roc_train, model2_roc_train, model1_roc_test, model2_roc_test

# base model1 part
def base_model1(n,data_train, data_valudation, data_test, label_train,label_valudation,label_test):
    alpha = 0.1
    data=np.concatenate((data_train,data_valudation),axis=0)
    data_train=np.concatenate((data,data_test),axis=0)
    label_train=np.concatenate((label_train,label_valudation),axis=0)
    label_temp=np.concatenate((label_train,np.repeat(0.,len(label_test)).reshape((len(label_test)),1)),axis=0)  
    y=intial_y(n,data_train,label_temp)  
    f=np.dot((1-alpha)*np.linalg.inv(np.eye(n)-laplace_matrix(n, data_train)),y)
    return f, data_train, label_train

def base_model1_roc(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
           data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test):
    f1, data1_train, label1_train=base_model1(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation,label1_test)
    f2, data2_train, label2_train=base_model1(n,data2_train, data2_valudation, data2_test, label2_train,label2_valudation,label2_test)
    base_roc1, base_roc2=model1_roc(n,f1,f2,data1_train,data2_train,label1_train,label1_test,label2_train,label2_test)
    return base_roc1, base_roc2
    

### model2 area
def adjance_matrix(data):
    v=data.shape[0]
    u=data.shape[1]
    left_matrix=np.eye(v,v)
    right_matrix=np.eye(u,u)
    left_matrix=np.diag(np.dot(data,np.repeat(1.,u).reshape((u,1)).ravel()))   
    right_matrix=np.diag(np.dot(np.repeat(1.,v).reshape((1,v)).ravel(),data))   
    left_matrix=np.linalg.inv(np.sqrt(left_matrix))
    right_matrix=np.linalg.inv(np.sqrt(right_matrix))
    adjance=np.dot(np.dot(left_matrix,data),right_matrix)
    return adjance.T

def intial_fv_fu(data1,data2,label1_train,label2_train):
    n=data1.shape[1]
    m1=data1.shape[0]
    m2=data2.shape[0]
    fv1=np.repeat(1./n,n).reshape((n,1))
    fv2=fv1.copy()
    fvc=fv1.copy()
    fu1=np.concatenate((label1_train,np.repeat(0.,m1-label1_train.shape[0]).reshape((m1-label1_train.shape[0],1))),axis=0) 
    fu2=np.concatenate((label2_train,np.repeat(0.,m2-label2_train.shape[0]).reshape((m2-label2_train.shape[0],1))),axis=0) 
    return fv1,fv2,fvc,fu1,fu2


def model2_update_fvi(n,adjance,fui,yvi,fvc,alpha,gamma):
    Y=(np.dot(adjance,fui)+alpha*yvi)/(1+alpha)-fvc
    model = linear_model.Lasso(1 / 2 * gamma)
    Q=np.eye(n)
    model.fit(Q, Y)
    fvi = model.coef_
    fvi=fvi.reshape((n,1))
    return fvi

def model2_update_fvc(n,adjance1,adjance2,fu1,fu2,fv1,fv2,yv1,yv2,alpha,gamma):
    Y=1/2*((np.dot(adjance1,fu1)+alpha*yv1)/(1+alpha)-fv1+(np.dot(adjance2,fu2)+alpha*yv2)/(1+alpha)-fv2)
    model = linear_model.Lasso(1 / 2 * gamma)
    Q=np.eye(n)
    model.fit(Q, Y)
    fvc = model.coef_
    fvc=fvc.reshape((n,1))
    return fvc

def model2_update_fui(n,adjance,fvi,fvc,alpha,yui):
    fui=(np.dot(adjance.T,fvi)+np.dot(adjance.T,fvc)+alpha*yui)/(1+alpha)
    return fui

def train_model(n,data1_train, data1_valudation, label1_train,label1_valudation, data2_train,\
                data2_valudation, label2_train, label2_valudation,alpha,gamma1,gamma2,gamma3):
    data1=np.concatenate((data1_train,data1_valudation),axis=0)
    data2=np.concatenate((data2_train,data2_valudation),axis=0)
    fv1,fv2,fvc,fu1,fu2=intial_fv_fu(data1,data2,label1_train,label2_train)
    label1_temp=np.concatenate((label1_train,np.repeat(0.,len(label1_valudation)).reshape((len(label1_valudation)),1)),axis=0)  
    label2_temp=np.concatenate((label2_train,np.repeat(0.,len(label2_valudation)).reshape((len(label2_valudation)),1)),axis=0)  
    yv1=intial_y(n,data1,label1_temp) 
    yv2=intial_y(n,data2,label2_temp) 
    adjance1=adjance_matrix(data1)
    adjance2=adjance_matrix(data2)
    yu1=fu1.copy()
    yu2=fu2.copy()
    for i in range(50):
        fv1=fv1.reshape((n,1))
        fv2=fv2.reshape((n,1))
        fvc=fvc.reshape((n,1))
        fv1_ori=fv1.copy()
        fv1=model2_update_fvi(n,adjance1,fu1,yv1,fvc,alpha,gamma1)
        fv2_ori=fv2.copy()
        fv2=model2_update_fvi(n,adjance2,fu2,yv2,fvc,alpha,gamma2)
        fvc_ori=fvc.copy()
        fvc=model2_update_fvc(n,adjance1,adjance2,fu1,fu2,fv1,fv2,yv1,yv2,alpha,gamma3)
        fu1=model2_update_fui(n,adjance1,fv1,fvc,alpha,yu1)
        fu2=model2_update_fui(n,adjance2,fv2,fvc,alpha,yu2)
        if np.sum(np.abs(fv1-fv1_ori))>100:
            break
        if np.sum(np.abs(fv1 - fv1_ori)) < 1e-5 or \
        np.sum(np.abs(fv2 - fv2_ori)) < 1e-5 or \
        np.sum(np.abs(fvc - fvc_ori)) < 1e-5:
            break
    fu1_test=fu1[len(fu1)-len(label1_valudation):]
    fpr, tpr, _ = roc_curve(label1_valudation.ravel(), fu1_test.ravel(), pos_label=-1)
    roc_auc1 = auc(fpr, tpr)
    fu2_test=fu2[len(fu2)-len(label2_valudation):]
    fpr, tpr, _ = roc_curve(label2_valudation.ravel(), fu2_test.ravel(), pos_label=-1)
    roc_auc2 = auc(fpr, tpr)
    return roc_auc1, roc_auc2
        
        
    
    
def test_model(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
           data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test,alpha,gamma1,gamma2,gamma3):
    data1=np.concatenate((data1_train,data1_test),axis=0)
    data2=np.concatenate((data2_train,data2_test),axis=0)
    label1_temp=np.concatenate((label1_train,np.repeat(0.,len(label1_test)).reshape((len(label1_test)),1)),axis=0)
    label2_temp=np.concatenate((label2_train,np.repeat(0.,len(label2_test)).reshape((len(label2_test)),1)),axis=0)
    fv1,fv2,fvc,fu1,fu2=intial_fv_fu(data1,data2,label1_train,label2_train)
    yv1=intial_y(n,data1,label1_temp)
    yv2=intial_y(n,data2,label2_temp)
    adjance1=adjance_matrix(data1)
    adjance2=adjance_matrix(data2)
    yu1=fu1.copy()
    yu2=fu2.copy()
    for i in range(50):
        fv1=fv1.reshape((n,1))
        fv2=fv2.reshape((n,1))
        fvc=fvc.reshape((n,1))
        fv1_ori=fv1.copy()
        fv1=model2_update_fvi(n,adjance1,fu1,yv1,fvc,alpha,gamma1)
        fv2_ori=fv2.copy()
        fv2=model2_update_fvi(n,adjance2,fu2,yv2,fvc,alpha,gamma2)
        fvc_ori=fvc.copy()
        fvc=model2_update_fvc(n,adjance1,adjance2,fu1,fu2,fv1,fv2,yv1,yv2,alpha,gamma3)
        fu1=model2_update_fui(n,adjance1,fv1,fvc,alpha,yu1)
        fu2=model2_update_fui(n,adjance2,fv2,fvc,alpha,yu2)
        if np.sum(np.abs(fv1-fv1_ori))>100:
            break
        if np.sum(np.abs(fv1 - fv1_ori)) < 1e-5 or \
        np.sum(np.abs(fv2 - fv2_ori)) < 1e-5 or \
        np.sum(np.abs(fvc - fvc_ori)) < 1e-5:
            break
    fu1_test=fu1[len(fu1)-len(label1_test):]
    fpr, tpr, _ = roc_curve(label1_test.ravel(), fu1_test.ravel(), pos_label=-1)
    roc_auc1 = auc(fpr, tpr)
    fu2_test=fu2[len(fu2)-len(label2_test):]
    fpr, tpr, _ = roc_curve(label2_test.ravel(), fu2_test.ravel(), pos_label=-1)
    roc_auc2 = auc(fpr, tpr)
    return roc_auc1, roc_auc2
    



def model2(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
           data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test,alpha,gamma1,gamma2,gamma3):
    train_roc_auc1, train_roc_auc2=train_model(n,data1_train, data1_valudation, label1_train,label1_valudation, data2_train,\
                                              data2_valudation, label2_train, label2_valudation,alpha,gamma1,gamma2,gamma3)
    test_roc_auc1, test_roc_auc2=test_model(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
                                            data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test,alpha,gamma1,gamma2,gamma3)
    return train_roc_auc1, train_roc_auc2, test_roc_auc1, test_roc_auc2
    
    
    


def model2_roc_all(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
                   data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test):
    alpha = 0.01
    gamma1_all=np.arange(5e-8,16e-8,2e-8)
    gamma2_all=np.arange(5e-8,16e-8,2e-8)
    gamma3_all=np.arange(5e-8,16e-8,2e-8)
    model1_roc_train=np.zeros((6,6,6))
    model2_roc_train=np.zeros((6,6,6))
    model1_roc_test=np.zeros((6,6,6))
    model2_roc_test=np.zeros((6,6,6))
    for iii in range(6):
        for jjj in range(6):
            for kkk in range(6):
                train_roc_auc1, train_roc_auc2, test_roc_auc1, test_roc_auc2=model2(n,data1_train, data1_valudation,\
                                                                                    data1_test, label1_train,label1_valudation,\
                                                                                    label1_test,data2_train, data2_valudation,\
                                                                                    data2_test, label2_train, label2_valudation,\
                                                                                    label2_test,alpha,gamma1_all[iii],gamma2_all[jjj],gamma3_all[kkk])
                model1_roc_train[iii,jjj,kkk]+=train_roc_auc1
                model2_roc_train[iii,jjj,kkk]+=train_roc_auc2
                model1_roc_test[iii,jjj,kkk]+=test_roc_auc1
                model2_roc_test[iii,jjj,kkk]+=test_roc_auc2
    return model1_roc_train, model2_roc_train, model1_roc_test, model2_roc_test



    
    
def base_model2(n,data_train, data_valudation, data_test,label_train,label_valudation, label_test):
    alpha = 0.01
    data_train=np.concatenate((data_train,data_valudation),axis=0)
    data=np.concatenate((data_train,data_test),axis=0)   
    label_train=np.concatenate((label_train,label_valudation),axis=0)
    fv=np.repeat(1/n,n).reshape((n,1))
    fu_ori=np.repeat(1/n,data.shape[0]).reshape((data.shape[0],1))
    fu=fu_ori.copy()
    y=intial_y(n,data_train,label_train)
    S=adjance_matrix(data)
    fu[:len(label_train)]=label_train
    for i in range(100):
        fv_ori=fv.copy()
        fv=(1-alpha)*y+alpha*np.dot(S,fu)
        fu=(1-alpha)*fu_ori+alpha*np.dot(S.T,fv)
        if np.sum(fv_ori-fv)<1e-5:
            break
    fu_test=fu[-len(label_test):]
    fpr, tpr, _ = roc_curve(label_test.ravel(), fu_test.ravel(), pos_label=-1)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def base_model2_roc(n,data1_train, data1_valudation, data1_test,\
               label1_train,label1_valudation, label1_test,\
               data2_train, data2_valudation,data2_test, label2_train, label2_valudation, label2_test):
    roc_auc1=base_model2(n,data1_train, data1_valudation, data1_test,label1_train,label1_valudation, label1_test)
    roc_auc2=base_model2(n,data2_train, data2_valudation, data2_test,label2_train,label2_valudation, label2_test)
    return roc_auc1, roc_auc2







### all models' results are got here 
def all_models(n,data1_train, data1_valudation, data1_test,\
               label1_train,label1_valudation, label1_test,\
               data2_train, data2_valudation,data2_test, label2_train, label2_valudation, label2_test):

    '''
# svc part
    svm_data1_temp_roc=svc_result(data1_train,data1_test,label1_train,label1_test)
    svm_data2_temp_roc=svc_result(data2_train,data2_test,label2_train,label2_test)
# model1 part
    model1_1_roc_train, model1_2_roc_train, model1_1_roc_test, model1_2_roc_test=model1_roc_all(n,data1_train, data1_valudation, data1_test,\
                                                                                        label1_train,label1_valudation, label1_test,\
                                                                                        data2_train, data2_valudation, data2_test,\
                                                                                        label2_train, label2_valudation, label2_test)
    loc1=np.argmax(model1_1_roc_train)
    model1_1_roc_train_max=np.max(model1_1_roc_train)
    final_model1_1_roc=model1_1_roc_test[loc1//9][loc1%9//3][loc1%9%3]
    loc2=np.argmax(model1_2_roc_train)
    model1_2_roc_train_max=np.max(model1_2_roc_train)
    final_model1_2_roc=model1_2_roc_test[loc2//9][loc2%9//3][loc2%9%3]
    
    
# base model1 part
    base_roc1, base_roc2=base_model1_roc(n,data1_train, data1_valudation, data1_test, label1_train,label1_valudation, label1_test,\
                                         data2_train, data2_valudation, data2_test, label2_train, label2_valudation, label2_test)

    '''
# model2 part
    model2_1_roc_train, model2_2_roc_train, model2_1_roc_test, model2_2_roc_test=model2_roc_all(n,data1_train, data1_valudation, data1_test,\
                                                                                                 label1_train,label1_valudation, label1_test,data2_train,\
                                                                                                 data2_valudation, data2_test, label2_train, label2_valudation,\
                                                                                                 label2_test)
    loc1=np.argmin(model2_1_roc_train)
    model2_1_roc_train_min=np.min(model2_1_roc_train)
    final_model2_1_roc=model2_1_roc_test[loc1//36][loc1%36//6][loc1%36%6]
    loc2=np.argmin(model2_2_roc_train)
    model2_2_roc_train_min=np.min(model2_2_roc_train)
    final_model2_2_roc=model2_2_roc_test[loc2//36][loc2%36//6][loc2%36%6]
# base model2 part
    base2_roc_auc1, base2_roc_auc2=base_model2_roc(n,data1_train, data1_valudation,\
                                       data1_test,label1_train,label1_valudation, label1_test,data2_train,\
                                       data2_valudation,data2_test, label2_train, label2_valudation, label2_test)
    
    '''
    return svm_data1_temp_roc, svm_data2_temp_roc, model1_1_roc_train_max, final_model1_1_roc, model1_2_roc_train_max, final_model1_2_roc, base_roc1,\
             base_roc2, model2_1_roc_train_max, final_model2_1_roc, model2_2_roc_train_max, final_model2_2_roc,base2_roc_auc1, base2_roc_auc2
    '''
    return model2_1_roc_train_min, final_model2_1_roc, model2_2_roc_train_min, final_model2_2_roc,base2_roc_auc1, base2_roc_auc2
    
    
    
    
    
    
    
    

    
### whole process
def whole_process(n,data_1_all,data_2_all,label_1_all,label_2_all,valudation_number,test_number,preprocess_scale):
### run 50 times, get the average
    svm_data1_temp_roc_all=0. 
    svm_data2_temp_roc_all=0.  
    final_model1_1_roc_all=0. 
    final_model1_2_roc_all=0. 
    base_roc1_all=0. 
    base_roc2_all=0. 
    final_model2_1_roc_all=0. 
    final_model2_2_roc_all=0.    
    model1_1_roc_train_max_all=0.
    model1_2_roc_train_max_all=0.
    model2_1_roc_train_max_all=0.
    model2_2_roc_train_max_all=0.
    base2_roc_auc1_all=0.
    base2_roc_auc2_all=0.   
    for i in range(50):
        data1_train, data1_valudation, data1_test, label1_train, label1_valudation,\
                  label1_test, data2_train, data2_valudation, data2_test, label2_train,\
                  label2_valudation, label2_test=cross_valudation(data_1_all,data_2_all,\
                  label_1_all,label_2_all,valudation_number,test_number)
        #svm_data1_temp_roc, svm_data2_temp_roc, model1_1_roc_train_max,\
        #final_model1_1_roc, model1_2_roc_train_max, final_model1_2_roc,\
        #base_roc1, base_roc2, model2_1_roc_train_max, final_model2_1_roc,\
        #model2_2_roc_train_max, final_model2_2_roc,base2_roc_auc1, base2_roc_auc2
        model2_1_roc_train_max, final_model2_1_roc, model2_2_roc_train_max, final_model2_2_roc,base2_roc_auc1, base2_roc_auc2=all_models(n,data1_train, data1_valudation, data1_test,\
                                                                                label1_train,label1_valudation, label1_test,\
                                                                                data2_train, data2_valudation,data2_test,\
                                                                                label2_train, label2_valudation, label2_test)
        '''
        svm_data1_temp_roc_all+=svm_data1_temp_roc
        svm_data2_temp_roc_all+=svm_data2_temp_roc
        model1_1_roc_train_max_all+=model1_1_roc_train_max
        final_model1_1_roc_all+=final_model1_1_roc
        model1_2_roc_train_max_all+=model1_2_roc_train_max
        final_model1_2_roc_all+=final_model1_2_roc
        base_roc1_all+=base_roc1
        base_roc2_all+=base_roc2
        '''
        model2_1_roc_train_max_all+=model2_1_roc_train_max
        final_model2_1_roc_all+=final_model2_1_roc
        model2_2_roc_train_max_all+=model2_2_roc_train_max
        final_model2_2_roc_all+=final_model2_2_roc
        base2_roc_auc1_all+=base2_roc_auc1
        base2_roc_auc2_all+=base2_roc_auc2
        print(i)
        temp_report=open('temp_report_swap_%s_new2.txt' %preprocess_scale,'a')
        print(file=temp_report)
        print('iterations:',i+1,file=temp_report)
        '''
        print(svm_data1_temp_roc_all/(i+1),file=temp_report)
        print(svm_data2_temp_roc_all/(i+1),file=temp_report)
        print(model1_1_roc_train_max_all/(i+1),file=temp_report)
        print(final_model1_1_roc_all/(i+1),file=temp_report)
        print(model1_2_roc_train_max_all/(i+1),file=temp_report)
        print(final_model1_2_roc_all/(i+1),file=temp_report)
        print(base_roc1_all/(i+1),file=temp_report)
        print(base_roc2_all/(i+1),file=temp_report)
        '''
        print(model2_1_roc_train_max,file=temp_report)
        print(final_model2_1_roc,file=temp_report)
        print(model2_2_roc_train_max,file=temp_report)
        print(final_model2_2_roc,file=temp_report)
        print(base2_roc_auc1,file=temp_report)
        print(base2_roc_auc2,file=temp_report)
        temp_report.close()
    final_report=open('final_report_swap_%s_new2.txt' %preprocess_scale,'w')
    '''
    print(svm_data1_temp_roc_all/(i+1),file=final_report)
    print(svm_data2_temp_roc_all/(i+1),file=final_report)
    print(model1_1_roc_train_max_all/(i+1),file=final_report)
    print(final_model1_1_roc_all/(i+1),file=final_report)
    print(model1_2_roc_train_max_all/(i+1),file=final_report)
    print(final_model1_2_roc_all/(i+1),file=final_report)
    print(base_roc1_all/(i+1),file=final_report)
    print(base_roc2_all/(i+1),file=final_report)
    '''
    print(model2_1_roc_train_max_all/(i+1),file=final_report)
    print(final_model2_1_roc_all/(i+1),file=final_report)
    print(model2_2_roc_train_max_all/(i+1),file=final_report)
    print(final_model2_2_roc_all/(i+1),file=final_report)
    print(base2_roc_auc1_all/(i+1),file=final_report)
    print(base2_roc_auc2_all/(i+1),file=final_report)
    final_report.close()
    
    
    
    
    
    
    
###  main area    
    
if __name__ == "__main__":
    preprocess_scale=False
    test_number=20
    valudation_number=20
    n,data_1_all,data_2_all,label_1_all,label_2_all=read_data(preprocess_scale)
    whole_process(n,data_1_all,data_2_all,label_1_all,label_2_all,valudation_number,test_number,preprocess_scale)
    
    
    
    
    
