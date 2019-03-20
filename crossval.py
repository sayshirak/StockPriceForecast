#-*- coding: UTF-8 -*-
__author__ = 'wqy'

import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

#path constants
featureRoot= "D:\\Project\\Python\\data\\20180919\\feature\\"
#labelfile = "D:\\Project\\Python\\20180919\\Face-example\\label\\label_yiyu.txt"
#labelfile = "D:\\Project\\Python\\20180919\\Face-example\\label\\label_jiaolv.txt"
#labelfile = "D:\\Project\\Python\\20180919\\Face-example\\label\\label_negitive.txt"
#labelfile = "D:\\Project\\Python\\20180919\\Face-example\\label\\label_positive.txt"
labelfile = "E:\\program\\PYTHON\\Face-example\\label\\label_zizun.txt"
#labelfile = "D:\\Project\\Python\\20180919\\Face-example\\label\\label_fengxian.txt"
#labelfile = "D:\\Project\\Python\\20180919\\Face-example\\label\\label_zhuangtai.txt"
#labelfile = "D:\\Project\\Python\\20180919\\Face-example\\label\\label_tezhi.txt"
#labelfile = "D:\\Project\\Python\\20180919\\Face-example\\label\\label_xiaoneng.txt"

sexfile = "E:\\program\\PYTHON\\Face-example\\label\\sex.txt"

#计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

#计算Pearson系数
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)   #计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

def processX(X,Y,label_threshold):
    num_label = Y.shape[0]
    #print num_label

    zero_X = []
    zero_Y = []
    one_X = []
    one_Y = []
    for index in range(num_label):
        if(Y[index] > label_threshold):
            if one_X == []:
                one_X = X[index,:]
                one_Y = Y[index]
            else:
                one_X = np.row_stack((one_X, X[index,:]))
                one_Y = np.row_stack((one_Y, Y[index]))
        else:
            if zero_X == []:
                zero_X = X[index,:]
                zero_Y = Y[index]
            else:
                zero_X = np.row_stack((zero_X, X[index,:]))
                zero_Y = np.row_stack((zero_Y, Y[index]))
    return zero_X,zero_Y,one_X,one_Y

def MaxMinNormalization(X):
    newX = X
    width = X.shape[1]
    height = X.shape[0]
    for i in range(width):
        minvalue = min(X[:,i])
        maxvalue = max(X[:,i])
        for j in range(height):
            newX[j,i] = (X[j,i]-minvalue)/(maxvalue-minvalue)

    return newX

def Z_ScoreNormalization(X):
    newX = X
    width = X.shape[1]
    height = X.shape[0]
    for i in range(width):
        avrvalue = np.average(X[:,i])
        stdvalue = np.std(X[:,i])
        for j in range(height):
            newX[j,i] = (X[j,i]-avrvalue)/stdvalue

    return newX

def sigmoid(X):
    newX = X
    width = X.shape[1]
    height = X.shape[0]
    for i in range(width):
        for j in range(height):
            newX[j,i] = 1.0 / (1 + np.exp(-float(X[j,i])))
    return newX

# 不根据label分组分组
def train_SVM(X, Y, sex, kernel, coef, gamma, degree):
    cv = model_selection.ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    p_all = np.ones((5,1), dtype=float)
    p1_all = np.ones((5,1), dtype=float)
    p0_all = np.ones((5,1), dtype=float)

    if kernel == 1:
        clf = SVR(kernel='rbf', C=coef, gamma=gamma)

    elif kernel == 2:
        clf = SVR(kernel='poly', C=coef, gamma=gamma, degree=degree)

    elif kernel == 3:
        clf = SVR(kernel='sigmoid', C=coef, gamma=gamma)

    elif kernel == 4:
        clf = LinearSVR(epsilon=0.0, tol=0.0001, C=coef, loss='epsilon_insensitive',
                    fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=1, max_iter=10000)

    ind = 0
    for train_index,test_index in cv.split(X):
       trainX = X[train_index]
       trainY = Y[train_index]

       test_x = X[test_index]
       testY = Y[test_index]

       clf.fit(trainX,trainY)

       preY = clf.predict(test_x)

       test_sex = sex[test_index]
       index1 = np.where(test_sex==1)
       index0 = np.where(test_sex==0)

       testY_1 = testY[index1]
       testY_0 = testY[index0]

       preY_1 = preY[index1]
       preY_0 = preY[index0]

       p = calcPearson(preY,testY)
       p_all[ind] = p

       p1 = calcPearson(preY_1,testY_1)
       p1_all[ind] = p1
       p0 = calcPearson(preY_0,testY_0)
       p0_all[ind] = p0

       ind = ind + 1
    return p_all, p1_all, p0_all

# 将大于阈值和小于阈值的分开
def train_SVM2(zero_X, zero_Y, one_X, one_Y, sex, kernel, coef, gamma, degree):

    cv = model_selection.ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    p_all = np.ones((5,1), dtype=float)
    p1_all = np.ones((5,1), dtype=float)
    p0_all = np.ones((5,1), dtype=float)

    if kernel == 1:
        clf = SVR(kernel='rbf', C=coef, gamma=gamma)

    elif kernel == 2:
        clf = SVR(kernel='poly', C=coef, gamma=gamma, degree=degree)

    elif kernel == 3:
        clf = SVR(kernel='sigmoid', C=coef, gamma=gamma)

    elif kernel == 4:
        clf = LinearSVR(epsilon=0.0, tol=0.0001, C=coef, loss='epsilon_insensitive',
                    fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=1, max_iter=10000)

    one_train_index = []
    one_test_index = []
    for train_index,test_index in cv.split(one_X):
        if one_train_index == []:
            one_train_index = train_index
            one_test_index = test_index
        else:
            one_train_index = np.row_stack((one_train_index, train_index))
            one_test_index = np.row_stack((one_test_index, test_index))

    ind = 0
    preY_total = []
    testY_total = []
    for train_index,test_index in cv.split(zero_X):
       trainX = zero_X[train_index]
       trainY = zero_Y[train_index]
       trainX = np.row_stack((trainX, one_X[one_train_index[ind]]))
       trainY = np.row_stack((trainY, one_Y[one_train_index[ind]]))

       test_x = zero_X[test_index]
       testY = zero_Y[test_index]
       test_x = np.row_stack((test_x, one_X[one_test_index[ind]]))
       testY = np.row_stack((testY, one_Y[one_test_index[ind]]))

       clf.fit(trainX,trainY)

       preY = clf.predict(test_x)
       #print testY
       #print preY

       test_sex = sex[test_index]
       index1 = np.where(test_sex==1)
       index0 = np.where(test_sex==0)

       testY_1 = testY[index1]
       testY_0 = testY[index0]

       preY_1 = preY[index1]
       preY_0 = preY[index0]

       p = calcPearson(preY,testY)
       p_all[ind] = p

       p1 = calcPearson(preY_1,testY_1)
       p1_all[ind] = p1
       p0 = calcPearson(preY_0,testY_0)
       p0_all[ind] = p0

       ind = ind + 1
    return p_all, p1_all, p0_all

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    p_total = []
    # 利用全部数据训练模型，然后针对男女分别计算相关系数
    # 1:female 0:male
    sex = np.loadtxt(sexfile)
    index1 = np.where(sex==1)
    index0 = np.where(sex==0)
    #print np.logspace(-5, 2, 8)
    mean_filter_window = [3,5,7,9,11]
    for mfw in mean_filter_window:
        featurefile = os.path.join(featureRoot, "feature_diff_meanfilter_"+str(int(mfw)))
        print(featurefile)
        # load feature
        X = np.loadtxt(featurefile)
        # standard normalization
        scaler = preprocessing.StandardScaler().fit(X)
        # scaler = preprocessing.MinMaxScaler().fit(X)
        X = scaler.transform(X)
        print(X.shape)
        Y = np.loadtxt(labelfile)
        print(Y.shape)

        CopyX = X

        for num_com in range(45,60,5):
            pca = PCA(n_components=num_com)
            pca_model = pca.fit(CopyX)
            #modelname ='./model/pca'
            #joblib.dump(pca_model, modelname)
            X = pca_model.transform(CopyX)
            print(X.shape)

            for kernel in range(1,5,1):
                for coef in np.logspace(-2, 2, 5):
                    #ind_coef in range(7):
                    #coef = 0.0001 * pow(10,ind_coef)
                    for gamma in np.logspace(-5, 2, 8):
                        for degree in range(2,6,1):

                            #for label_threshold in range(25,26,1):

                                # 训练的时候能够更加平均分配
                                #zero_X,zero_Y,one_X,one_Y = processX(X, Y, label_threshold)
                                #p, p1, p0 = train_SVM2(zero_X, zero_Y, one_X, one_Y, sex, kernel, coef, gamma, degree)
                                p, p1, p0 = train_SVM(X, Y, sex, kernel, coef, gamma, degree)
                                #print p
                                mean_p = np.mean(p)
                                mean_p1 = np.mean(p1)
                                mean_p0 = np.mean(p0)
                                #print mean_p

                                step_result = [mfw, num_com, kernel, coef, gamma, degree, mean_p, mean_p1, mean_p0]

                                if p_total == []:
                                    p_total = step_result
                                else:
                                    p_total = np.row_stack((p_total, step_result))
    # 1/2: StandardScaler/MinMaxScaler   1/2: 分组/不分组
    np.savetxt('data/tmp1_2.txt',p_total)

    #p_all = pd.read_csv('data/tmp.txt',sep=" ",header=None)
    p_all = pd.DataFrame(p_total)
    #print p_all
    p_all = p_all.fillna(0)
    #print p_all.describe()
    p_all_data = p_all.loc[:,6]
    max_value = max(p_all_data)
    max_index = p_all[(max_value-p_all_data)<0.05].index.tolist()
    print(p_all.loc[max_index])
    print(max_value)
    #print "max value:" + str(np.max(p_all))
    p_all_data = p_all.loc[:,7]
    max_value = max(p_all_data)
    max_index = p_all[(max_value-p_all_data)<0.05].index.tolist()
    print(p_all.loc[max_index])
    print(max_value)

    p_all_data = p_all.loc[:,8]
    max_value = max(p_all_data)
    max_index = p_all[(max_value-p_all_data)<0.05].index.tolist()
    print(p_all.loc[max_index])
    print(max_value)


