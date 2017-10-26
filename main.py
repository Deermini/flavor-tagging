#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:06:43 2017

@author: Deermini
"""

from numpy import *
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
from GCForest import *
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import pickle
import collections


def roc_b_sig_c_bkg(likelyness,test_true):
    d=collections.Counter(test_true)
    b_total=d[0]
    c_total=d[1]
    q_total=d[2]
    b_cut=[]
    c_cut=[]
    q_cut=[]
    for i in range(100):
        ratio=float(i)/float(100)
        test_cut=test_true[likelyness[:,0]>=ratio]
        dd=collections.Counter(test_cut)
        bb=float(dd[0])/float(b_total)
        cc=1.0-float(dd[1])/float(c_total)
        qq=1.0-float(dd[2])/float(q_total)
        b_cut.append(bb)
        c_cut.append(cc)
        q_cut.append(qq)
    return b_cut,c_cut,q_cut
def roc_c_sig_b_bkg(likelyness,test_true):
    d=collections.Counter(test_true)
    b_total=d[0]
    c_total=d[1]
    q_total=d[2]
    b_cut=[]
    c_cut=[]
    q_cut=[]
    for i in range(100):
        ratio=float(i)/float(100)
        test_cut=test_true[likelyness[:,1]>=ratio]
        dd=collections.Counter(test_cut)
        bb=1.0-float(dd[0])/float(b_total)
        cc=float(dd[1])/float(c_total)
        qq=1.0-float(dd[2])/float(q_total)
        b_cut.append(bb)
        c_cut.append(cc)
        q_cut.append(qq)
    return b_cut,c_cut,q_cut

def AUC(a,b):
        length=len(a)
        auc_score=0
        for i in range(length-1):
            c=(b[i+1]-b[i])*(a[i]+a[i+1])/2.0
            auc_score+=c
        return auc_score
    
def run(train_X,test_X,train_y,test_y):
    """
	存储训练的模型及roc曲线的100个点方便以后画图
    """
    #GBDT model
    GBDT=GradientBoostingClassifier(n_estimators=200,subsample=0.9,learning_rate=0.3,min_samples_leaf=5,random_state=33)
    GBDT.fit(train_X,train_y)
    pickle.dump(GBDT,open('./model/GBDT.pickle.dat','wb'))
  
    GBDT=pickle.load(open('./model/GBDT.pickle.dat','rb'))
    y_score=GBDT.predict_proba(test_X)
    
    #XGBoost model
    xgb_model = xgb.XGBClassifier(max_depth=9, 
    				  min_child_weight=1,
    				  subsample=0.9,
    				  colsample_bytree=0.8,
    				  learning_rate = 0.1,
    				  n_estimators = 200,
    								  )
    xgb_model=fit(train_X, train_y)
    pickle.dump(xgb_model,open('./model/xgb_model.pickle.dat','wb'))
    
    xgb_model=pickle.load(open('./model/xgb_model.pickle.dat','rb'))
    y_score1= xgb_model.predict_proba(test_X)

    #gcforest
    gcf = gcForest(tolerance=0.0,n_cascadeRFtree=300,n_jobs=-1)
    _ = gcf.cascade_forest(train_X,train_y)
    joblib.dump(gcf, 'F:/project files/FlavorTagging/model/gcforest.pkl')

    gcf=joblib.load('F:/project files/FlavorTagging/model/gcforest.pkl')
    pred_proba = gcf.cascade_forest(test_X)
    y_score2 = np.mean(pred_proba, axis=0)
	
    #存GBDT下ROC曲线的100个点
    b_roc_GBDT=np.array(roc_b_sig_c_bkg(y_score,test_y)).T
    c_roc_GBDT=np.array(roc_c_sig_b_bkg(y_score,test_y)).T
    np.savetxt('roc/b_roc_GBDT.txt',b_roc_GBDT,delimiter=",")
    np.savetxt('roc/c_roc_GBDT.txt',c_roc_GBDT,delimiter=",")
    
    #存xgboost下ROC曲线的100个点
    b_roc_xgb=np.array(roc_b_sig_c_bkg(y_score1,test_y)).T
    c_roc_xgb=np.array(roc_c_sig_b_bkg(y_score1,test_y)).T
    np.savetxt('roc/b_roc_xgb.txt',b_roc_xgb,delimiter=",")
    np.savetxt('roc/c_roc_xgb.txt',c_roc_xgb,delimiter=",")

    #存gcforest下ROC曲线的100个点
    b_roc_gcforest=np.array(roc_b_sig_c_bkg(y_score2,test_y)).T
    c_roc_gcforest=np.array(roc_c_sig_b_bkg(y_score2,test_y)).T
    np.savetxt('roc/b_roc_gcforest.txt',b_roc_gcforest,delimiter=",")
    np.savetxt('roc/c_roc_gcforest.txt',c_roc_gcforest,delimiter=",")

def run1():
    """
	运行完run函数后，我们就可以画出几条相应的曲线并求出AUC值
    """
	
    #载入文件
    b_roc_GBDT=np.loadtxt('roc/b_roc_GBDT.txt',delimiter=',')
    c_roc_GBDT=np.loadtxt('roc/c_roc_GBDT.txt',delimiter=',')
    b_roc_xgb=np.loadtxt('roc/b_roc_xgb.txt',delimiter=',')
    c_roc_xgb=np.loadtxt('roc/c_roc_xgb.txt',delimiter=',')
    b_roc_gcforest=np.loadtxt("roc/b_roc_gcforest.txt",delimiter=",")
    c_roc_gcforest=np.loadtxt("roc/c_roc_gcforest.txt",delimiter=",")
	
    #画图并计算AUC值
    #0和1类进行计算，0为正类
    plt.subplot(221)
    roc_auc=AUC(b_roc_xgb[:,0],b_roc_xgb[:,1])
    roc_auc1=AUC(b_roc_GBDT[:,0],b_roc_GBDT[:,1])
    roc_auc2=AUC(b_roc_gcforest[:,0],b_roc_gcforest[:,1])
    plt.plot(b_roc_xgb[:,0],b_roc_xgb[:,1],c='green',label='xgboost (area = {0:0.5f})'''.format(roc_auc))
    plt.plot(b_roc_GBDT[:,0],b_roc_GBDT[:,1],c='red',label='GBDT (area = {0:0.5f})'''.format(roc_auc1))
    plt.plot(b_roc_gcforest[:,0],b_roc_gcforest[:,1],c='blue',label='gcforest (area = {0:0.5f})'''.format(roc_auc2))
    plt.legend(loc="lower left")
    
    #0和2类进行计算，0为正类
    plt.subplot(222)
    roc_auc=AUC(b_roc_xgb[:,0],b_roc_xgb[:,2])
    roc_auc1=AUC(b_roc_GBDT[:,0],b_roc_GBDT[:,2])
    roc_auc2=AUC(b_roc_gcforest[:,0],b_roc_gcforest[:,2])
    plt.plot(b_roc_xgb[:,0],b_roc_xgb[:,2],c='green',label='xgboost (area = {0:0.5f})'''.format(roc_auc))
    plt.plot(b_roc_GBDT[:,0],b_roc_GBDT[:,2],c='red',label='GBDT (area = {0:0.5f})'''.format(roc_auc1))
    plt.plot(b_roc_gcforest[:,0],b_roc_gcforest[:,2],c='blue',label='gcforest (area = {0:0.5f})'''.format(roc_auc2))
    plt.legend(loc="lower left")
    
    #1和0类进行计算，1为正类
    plt.subplot(223)
    roc_auc=AUC(c_roc_xgb[:,1],c_roc_xgb[:,0])
    roc_auc1=AUC(c_roc_GBDT[:,1],c_roc_GBDT[:,0])
    roc_auc2=AUC(c_roc_gcforest[:,1],c_roc_gcforest[:,0])
    plt.plot(c_roc_xgb[:,1],c_roc_xgb[:,0],c='green',label='xgboost (area = {0:0.5f})'''.format(roc_auc))
    plt.plot(c_roc_GBDT[:,1],c_roc_GBDT[:,0],c='red',label='GBDT (area = {0:0.5f})'''.format(roc_auc1))
    plt.plot(c_roc_gcforest[:,1],c_roc_gcforest[:,0],c='blue',label='gcforest (area = {0:0.5f})'''.format(roc_auc2))
    plt.legend(loc="lower left")
    
    #1和2类进行计算,1为正类
    plt.subplot(224)
    roc_auc=AUC(c_roc_xgb[:,1],c_roc_xgb[:,2])
    roc_auc1=AUC(c_roc_GBDT[:,1],c_roc_GBDT[:,2])
    roc_auc2=AUC(c_roc_gcforest[:,1],c_roc_gcforest[:,2])
    plt.plot(c_roc_xgb[:,1],c_roc_xgb[:,2],c='green',label='xgboost (area = {0:0.5f})'''.format(roc_auc))
    plt.plot(c_roc_GBDT[:,1],c_roc_GBDT[:,2],c='red',label='GBDT (area = {0:0.5f})'''.format(roc_auc1))
    plt.plot(c_roc_gcforest[:,1],c_roc_gcforest[:,2],c='blue',label='gcforest (area = {0:0.5f})'''.format(roc_auc2))
    plt.legend(loc="lower left")
    
    plt.show()
    

def run2(train_X,test_X,train_y,test_y):
    """采用一对多的方式"""
    from sklearn.metrics import roc_curve, auc
    
    GBDT=pickle.load(open('./model/GBDT.pickle.dat','rb'))
    y_score=GBDT.predict_proba(test_X)
    fpr, tpr, _ = roc_curve(test_y,y_score[:,1],pos_label=1)
    roc_auc= auc(fpr, tpr)
    print(roc_auc)
    
    xgb_model=pickle.load(open('./model/xgb_model.pickle.dat','rb'))
    y_score2= xgb_model.predict_proba(test_X)
    fpr2, tpr2, _ = roc_curve(test_y,y_score2[:,1],pos_label=1)
    roc_auc2= auc(fpr2, tpr2)
    print(roc_auc2)
    

def run3(train_X,test_X,train_y,test_y):
    """
    使用经过rfe进行特征排序后的结果计算不同特征下的AUC值
    """
    """载入计算好的特征重要性"""
    feature_importance=np.loadtxt("feature_selection_result/ranking.txt")
    a=np.argsort(feature_importance)
    AUC=[]
    
    #采用一对一的方式对数据进行计算，抽出数据中相应的类别。
    index_train=list(where(train_y==0)[0]);index_train1=list(where(train_y==1)[0])
    index_train.extend(index_train1)
    train_X=train_X[index_train,];train_y=train_y[index_train]
    
    index_test = list(where(test_y==0)[0]);index_test1 = list(where(test_y==1)[0])
    index_test.extend(index_test1)
    test_X = test_X[index_test];test_y = test_y[index_test]
    
    #选取特征数目从10到63
    for i in range(10,63,1):
        index=list(a[:i])
        train_X_new = train_X[:, index]
        test_X_new=test_X[:,index]
        xgb_model = xgb.XGBClassifier(max_depth=9, 
    				      min_child_weight=1,
    				      subsample=0.9,
    				      colsample_bytree=0.8,
    		       		      learning_rate = 0.1,
    				      n_estimators = 200,
    					 )
        xgb_model.fit(train_X_new,train_y)
        y_score=xgb_model.predict_proba(test_X_new)
        fpr, tpr, _ = roc_curve(test_y,y_score[:,0],pos_label=0)
        roc_auc= auc(fpr, tpr)
        AUC.append(roc_auc)
        print(i,roc_auc)
    
    plt.figure()
    plt.plot(range(10,63,1),AUC,label='rfe-xgboost')
    plt.xlabel("auc")
    plt.ylabel("the number of features")
    plt.title('ROC (1-3)')
    plt.legend(loc=0)
    plt.show()
    
if __name__ == '__main__':
	
    #载入数据
    train_data = np.load("data/data_train.npy")
    test_data = np.load("data/data_test.npy")
    train_X, test_X = train_data[:, 1:], test_data[:, 1:]
    train_y, test_y = train_data[:, 0], test_data[:, 0]
    #先运行run()得到模型，再运行run1()即可画出图形
    run(train_X,test_X,train_y,test_y)
    run1()



