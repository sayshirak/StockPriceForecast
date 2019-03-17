#-*- coding: UTF-8 -*-
__author__ = 'gy'
#获取用于计算模型的历史数据
import numpy as np
import pandas as pd
from rqalpha.api import *
import datetime

#只能在米筐平台上运行，这里作为备份

#get_price('中证500', start_date='2016-02-03', end_date='2016-02-13', frequency='1m', fields=None)
df = pd.DataFrame(get_price('中证500', start_date='2007-01-15', end_date=datetime.date.today(), frequency='1d', fields=['open','close','high','low','volume','total_turnover']))
dfPEPB = pd.DataFrame(index_indicator('中证500', start_date='2000-02-03', end_date=datetime.date.today()))
#将dfPEPB中交易日期列作为索引
dfPEPB.set_index(["trade_date"], inplace=True)
#将dfPEPB中PE和PB两行加到df后面
df['pb'] = dfPEPB['pb']
df['pe_ttm'] = dfPEPB['pe_ttm']
dfTomorrow = pd.DataFrame(df['close'])
#删除df第一行
df = df.drop(index=[df.index[len(df)-1]])
df.to_csv('ZZ500.csv')
#删除dfTomorrow最后一行
#预测算法是用df的数据预测dfTomorrow的中证500指数
dfTomorrow =dfTomorrow.drop(index=[dfTomorrow.index[0]])
dfTomorrow.to_csv('ZZ500Tomorrow.csv')

