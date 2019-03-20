#-*- coding: UTF-8 -*-
__author__ = 'gy'
'''
直观上来讲 SVM 分类（SVC Support Vector Classification）与 SVR（Support Vector Regression）的区别如下：
分类是找一个平面，使得边界上的点到平面的距离最远，回归是让每个点到回归线的距离最小。
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import crossval as cr
import time
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
svr = SVR(kernel = 'linear',C = 1e3,gamma = 0.01)
#svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma=0.1,kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

def createModel():

    # 导入数据
    #ZZ500.csv存放了上1个交易日之前(包括上1个交易日)的数据
    #ZZ500Tomorrow.csv存放了本交易日之前(包括本交易日)的收盘价
    X = pd.read_csv('ZZ500.csv', delimiter=',').iloc[:,1:]
    Y = pd.read_csv('ZZ500Tomorrow.csv', delimiter=',').iloc[:,1:]
    X, Y = checkDat(X,Y)
    X = X.values[:,:]
    Y = Y.values[:,:]


    # 从数据集中取80%作为测试集，其他作为训练集
    # 从数据集中取80%作为测试集，其他作为训练集
    height = int(X.shape[0]*0.1)
    # 创建SVR模型
    print("SVR建模")
    trainSize = int(0.9*height)
    testSize = int(height - trainSize)
    X_train = X[0:trainSize,:]
    X_test = X[trainSize:height,:]
    Y_train = Y[0:trainSize,:]
    Y_test = Y[trainSize:height,:]


    # 用训练集训练模型
    #print("%d start-------"%(height))
    #startTime = time.time()
    svr.fit(X_train, Y_train.ravel())
    #endTime = time.time()
    #durationTime = endTime - startTime
    #print("%d end" % (height))
    #print("duration: %s"%(durationTime))

    # 用训练得出的模型进行预测
    diabetes_y_pred = svr.predict(X_test)

    verifyModel(diabetes_y_pred,X_test,Y_test)


#校验数据，如果有空的情况，将前一个交易日和后一个交易日做算术平均写入空单元格
def checkDat(X,Y):
    widthX = X.shape[1]
    widthY = Y.shape[1]
    height = X.shape[0]
    for i in range(height):
        for j in range(widthX):
            if np.isnan(X.iloc[i,j]):
                if 0 == i:
                    X.iloc[i,j] = X.iloc[i+1,j]
                elif ( height-1 == i ) and ( height > 1 ):
                    X.iloc[i,j] = X.iloc[i-1,j]
                elif ( 1 == height ):
                    X.iloc[i, j] = 0
                else:
                    X.iloc[i,j] = ( X.iloc[i+1,j] + X.iloc[i-1,j] )/2

    return X,Y

# 校验模型精度
def verifyModel(y_pred,X_test, Y_test):
    '''
    preTest = [5557.1556,5424.8642,5557.1556,5388.2464,22434923600,201483723055]
    ZZIndex = 0
    for i in range(len(preTest)):
        ZZIndex = ZZIndex + preTest[i]*regr.coef_[i]
    '''
    # 将预测准确率打印出来
    predict = np.array(y_pred)
    true = np.array(Y_test)
    Ac = accuracy(predict, true)  # 判断指数准确率
    print("判断指数准确率=", Ac * 100, '%')
    Ac = accuracy2(predict, true)  # 判断涨跌准确率
    print("判断涨跌准确率=", Ac * 100, '%')

    # 以 R-Squared 对预测准确率进行计算，将其打印出来
    print("R-Squared Accuracy=", (svr.score(X_test, Y_test)) * 100, '%')

    #计算皮尔逊相关系数，需要分别计算决策风格的5个维度的相关
    pearson = cr.calcPearson(Y_test, y_pred)
    print('Pearson correlation coefficient: %2f' % pearson)
    # 将测试结果以图标的方式显示出来
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'go-', label="predict value")
    plt.plot(range(len(y_pred)), Y_test, 'ro-', label="true value")
    plt.legend()
    plt.show()

# 以 预测准确率=（预测正确样本数）/（总测试样本数）* 100% 对预测准确率进行计算，设定 ErrorTolerance = 1%
def accuracy(predict, true):
    sizeofall = len(true)
    sizeofright = 0
    for i in range(0, sizeofall):
        est = abs(predict[i] - true[i]) / true[i]
        if est < 0.01:
            sizeofright = sizeofright + 1

    return sizeofright/sizeofall

#另一种计算预测准确率的方法，当日真实值和预测值和上一个交易日的真实值做减法，如果都为正或都为负则预测成功，否则失败
def accuracy2(predict, true):
    sizeofall = len(true)
    sizeofright = 0
    for i in range(1, sizeofall):
        if(((predict[i] - true[i-1]>0) and (true[i] - true[i-1]>0))
            or ((predict[i] - true[i-1]>0) and (true[i] - true[i-1]<0))):
            sizeofright = sizeofright + 1

    return sizeofright/sizeofall

def example():
    n_samples, n_features = 3000, 24
    np.random.seed(0)
    Y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)
    height = len(X)
    trainSize = int(0.8*height)
    testSize = height - trainSize
    X_train = X[0:trainSize]
    X_test = X[trainSize:height]
    Y_train = Y[0:trainSize]
    Y_test = Y[trainSize:height]
    #clf = SVR(gamma=0.1, C=1.0, epsilon=0.2)
    svr.fit(X_train, Y_train)
    diabetes_y_pred = svr.predict(X_test)
    verifyModel(diabetes_y_pred,X_test,Y_test)


if __name__ == '__main__':
    time1 = time.time()
    createModel()
    #example()
    time2 = time.time()