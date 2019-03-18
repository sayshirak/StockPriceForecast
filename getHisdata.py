import numpy as np
import pandas as pd
import datetime

#这是米筐研究平台的备份，不能离线运行
param = ['open', 'close', 'high', 'low', 'volume', 'total_turnover']
# get_price('中证500', start_date='2016-02-03', end_date='2016-02-13', frequency='1m', fields=None)
df = pd.DataFrame(
    get_price('中证500', start_date='2007-01-15', end_date=datetime.date.today(), frequency='1d', fields=param))
dfPEPB = pd.DataFrame(index_indicator('中证500', start_date='2000-02-03', end_date=datetime.date.today()))
dfSZZS = pd.DataFrame(
    get_price('上证指数', start_date='2007-01-15', end_date=datetime.date.today(), frequency='1d', fields=param))
dfHS300 = pd.DataFrame(
    get_price('沪深300', start_date='2007-01-15', end_date=datetime.date.today(), frequency='1d', fields=param))
dfSZCZ = pd.DataFrame(
    get_price('深证成指', start_date='2007-01-15', end_date=datetime.date.today(), frequency='1d', fields=param))
# 将dfPEPB中交易日期列作为索引
dfPEPB.set_index(["trade_date"], inplace=True)

# 将dfPEPB中PE和PB两行加到df后面
df['pb'] = dfPEPB['pb']
df['pe_ttm'] = dfPEPB['pe_ttm']

# 将上证指数的历史数据加到df后面
for i in range(len(param)):
    paramAdd = 'SZZS' + param[i]
    df[paramAdd] = dfSZZS.iloc[:, i]

# 将沪深300的历史数据加到df后面
for i in range(len(param)):
    paramAdd = 'HS300' + param[i]
    df[paramAdd] = dfHS300.iloc[:, i]

# 将深圳成指的历史数据加到df后面
for i in range(len(param)):
    paramAdd = 'SZCZ' + param[i]
    df[paramAdd] = dfSZCZ.iloc[:, i]

# print(df)

dfTomorrow = pd.DataFrame(df['close'])
# 删除df第一行
df = df.drop(index=[df.index[len(df) - 1]])
df.to_csv('ZZ500.csv')
# 删除dfTomorrow最后一行
# 预测算法是用df的数据预测dfTomorrow的中证500指数
dfTomorrow = dfTomorrow.drop(index=[dfTomorrow.index[0]])
dfTomorrow.to_csv('ZZ500Tomorrow.csv')

