#-*- coding: UTF-8 -*-
__author__ = 'gy'
#获取用于计算模型的历史数据
import numpy as np
import pandas as pd
from rqalpha.api import *


import datetime

index = "000905.XSHG"
nextDate = datetime.date.today()+datetime.timedelta(days=4)
forecastTradeDate = get_previous_trading_date(nextDate)

model = [ -1.29844591e-01,1.23706203e+00,-3.76624367e-02,-7.21035534e-02,4.93911638e-09,-2.02546416e-10,-6.31834722e+00,2.16241357e-01]
lastTradeDate = get_previous_trading_date(forecastTradeDate)
print("上一个交易日:%s"%(lastTradeDate))
print("下一个交易日:%s"%(forecastTradeDate))
data = get_price(index, start_date=lastTradeDate, end_date=lastTradeDate, fields = ['open','close','high','low','volume','total_turnover']).iloc[0:,0:6]
#print(data)
dataTotal = []
for i in range(data.shape[1]):
    #print(data.iloc[0:,i:i+1])
    dataTotal.append(data.values[0:,i:i+1])
getPBPE = index_indicator('中证500', start_date=lastTradeDate, end_date=lastTradeDate).iloc[:,1:3]
print(getPBPE)
for i in range(getPBPE.shape[1]):
    dataTotal.append(getPBPE.values[0:,i:i+1])
price = 0
#for i in range(len(model)):
    #print(model[i])
    #print(dataTotal[i][0])
    #price = price + model[i]*dataTotal[i][0]
#print(data)
#print(getPBPE)
#print(price)