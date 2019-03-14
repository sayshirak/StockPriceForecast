#-*- coding: UTF-8 -*-
__author__ = 'gy'
#滑动窗口取特征值
import os
import time
import csv
import numpy as np
import scipy
import pandas as pd

#path constants
pathInputRoot = "F:\\program\\decisionMaking\\FaceData-SelectFP"
pathOutputRootFFT = "F:\\program\\decisionMaking\\FaceData-SlideWindowFFT"
pathOutputRootStatistical = "F:\\program\\decisionMaking\\FaceData-SlideWindowStatistical"
win = 128   #窗口长度，一般为2的n次方，这样FFT的结果比较准
step = 64  #滑动步长
point = 100 #面部数据点数

'''
计算每一列所有窗口的fft同一个序号的系数的均值和方差
'''
def calFFTMeanVar(fft):
    meanVar = []
    mean = []
    var = []
    #每一行是一个窗口，每个窗口32个点
    fft = pd.DataFrame(fft)
    width = fft.shape[1]
    height = fft.shape[0]
    for i in range(width):
        mean.append(np.mean(fft[i]))
        var.append(np.var(fft[i]))
    meanVar.append(mean)
    meanVar.append(var)
    return meanVar

'''
对每一列每个窗口采用FFT，取前32个系数，然后对所有窗口同样序号的FFT系数取均值或方差作为特征值
这样每一列获得32*2=64个特征
FFT的结果是一个复数，有实部和虚部，系数就是求复数的模。例如复数
z = x+iy，则z的模
|z|=√(x^2+y^2)
'''
def getSlideFilesFFT(pathFile,file):
    data = np.loadtxt(pathFile,delimiter=',')
    height = data.shape[0]
    width = data.shape[1]
    outFileIdx = 0
    fftFileTotal = []
    for col in range(width):
        #获取每一个窗口的fft，存放在fftWin
        fftWin = []
        #每一列的特征放在fftCol
        fftCol = []
        for row in range(0,height-step,step):
            '''
            FFT的结果第1个数最大，剩下的数以中数为中心对称。这是傅里叶变换所决定的
            '''
            if row+win <= height:
                fft = scipy.fft(data[row:row+win,col])[0:32]
            elif height - row >= 32:
                fft = scipy.fft(data[row:height,col])[0:32]
            #print(scipy.fft(data[row:height,col]))
            modulo = []
            #求每一个复数的模
            for i in range(len(fft)):
                modulo.append(np.sqrt(np.square(fft[i].real)+np.square(fft[i].imag)))
            fftWin.append(modulo)
        meanVar = calFFTMeanVar(fftWin)
        for j in range(len(meanVar)):
            fftCol.extend(meanVar[j])
        fftFileTotal.extend(fftCol)
    return fftFileTotal




def slideWindow():
    fileList = os.listdir(pathInputRoot)
    getFeature = []
    for file in range(len(fileList)):
        pathFile = os.path.join(pathInputRoot, str(file)+".csv")
        print("load " + pathFile)
        #print pathOutputFile
        fftFile = getSlideFilesFFT(pathFile,file)
        getFeature.append(fftFile)
    outFile = os.path.join(pathOutputRootFFT, "fftFeature.csv")
    np.savetxt(outFile, getFeature, fmt='%.18e', delimiter=',')

if __name__ == '__main__':
    time1 = time.time()

    slideWindow()

    time2 = time.time()