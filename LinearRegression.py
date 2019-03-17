#-*- coding: UTF-8 -*-
__author__ = 'gy'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

regr = linear_model.LinearRegression()


def createModel():
    # 导入数据
    #ZZ500.csv存放了上1个交易日之前(包括上1个交易日)的数据
    #ZZ500Tomorrow.csv存放了本交易日之前(包括本交易日)的收盘价
    X = pd.read_csv('ZZ500.csv', delimiter=',').iloc[:,1:]
    Y = pd.read_csv('ZZ500Tomorrow.csv', delimiter=',').iloc[:,1:]
    X, Y = checkDat(X,Y)
    X = X.values[:,:]
    Y = Y.values[:,:]
    #used_features = ["high", "low", "open", "volume","total_turnover"]
    #X = data[used_features]
    #y = data["close"]

    # 从数据集中取70%作为测试集，其他作为训练集
    height = X.shape[0]
    trainSize = int(0.7*height)
    testSize = height - trainSize
    X_train = X[0:trainSize,:]
    X_test = X[trainSize:height,:]
    Y_train = Y[0:trainSize,:]
    Y_test = Y[trainSize:height,:]
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=0,
    )
    '''
    # 创建线性回归模型
    print("线性回归建模")
    # 用训练集训练模型
    regr.fit(X_train, Y_train)
    # 用训练得出的模型进行预测
    diabetes_y_pred = regr.predict(X_test)
    # 输出回归方程
    print('Coefficients: \n', regr.coef_)
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
def verifyModel(diabetes_y_pred,X_test, Y_test):
    '''
    preTest = [5557.1556,5424.8642,5557.1556,5388.2464,22434923600,201483723055]
    ZZIndex = 0
    for i in range(len(preTest)):
        ZZIndex = ZZIndex + preTest[i]*regr.coef_[i]
    '''
    # 将预测准确率打印出来
    predict = np.array(diabetes_y_pred)
    true = np.array(Y_test)
    Ac = accuracy(predict, true)  # 判断指数准确率
    print("判断指数准确率=", Ac * 100, '%')
    Ac = accuracy2(predict, true)  # 判断涨跌准确率
    print("判断涨跌准确率=", Ac * 100, '%')

    # 以 R-Squared 对预测准确率进行计算，将其打印出来
    print("R-Squared Accuracy=", (regr.score(X_test, Y_test)) * 100, '%')

    # 将测试结果以图标的方式显示出来
    plt.figure()
    plt.plot(range(len(diabetes_y_pred)), diabetes_y_pred, 'go-', label="predict value")
    plt.plot(range(len(diabetes_y_pred)), Y_test, 'ro-', label="true value")
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


if __name__ == '__main__':
    time1 = time.time()
    createModel()
    time2 = time.time()