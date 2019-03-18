import numpy as np
import pandas as pd
import datetime

#这是米筐研究平台的备份，不能离线运行
param = ['open', 'close', 'high', 'low', 'volume', 'total_turnover']
index = ['中证500', '上证指数', '沪深300', '深证成指']
nextDate = datetime.date.today() + datetime.timedelta(days=1)
forecastTradeDate = get_previous_trading_date(nextDate)

model = [2.62232760e-02, 1.42155210e+00, -2.24324074e-01, -2.19099673e-01, 4.25569587e-09, -1.92310607e-11,
         -1.51339628e+01,
         2.26252804e-01, 5.84562498e-01, -1.56973869e+00, 9.79394040e-01, -1.12369794e-02, 7.02668324e-09,
         -7.57556573e-10,
         -1.32911467e+00, 1.28494680e+00, -3.02247333e-01, 3.88361171e-01, -1.17489467e-08, 1.08212284e-09,
         1.01076541e-01,
         -1.32325859e-01, 8.58795417e-03, 1.45597842e-02, -5.56628876e-09, 2.06253681e-10]
lastTradeDate = get_previous_trading_date(forecastTradeDate)
print("上一个交易日:%s" % (lastTradeDate))
print("下一个交易日:%s" % (forecastTradeDate))
df = pd.DataFrame(get_price(index[0], start_date=lastTradeDate, end_date=lastTradeDate, fields=param)).iloc[0:, 0:6]
dfSZZS = pd.DataFrame(get_price(index[1], start_date=lastTradeDate, end_date=lastTradeDate, fields=param)).iloc[0:, 0:6]
dfHS300 = pd.DataFrame(get_price(index[2], start_date=lastTradeDate, end_date=lastTradeDate, fields=param)).iloc[0:,
          0:6]
dfSZCZ = pd.DataFrame(get_price(index[3], start_date=lastTradeDate, end_date=lastTradeDate, fields=param)).iloc[0:, 0:6]
# print(data)

dataTotal = []
for i in range(df.shape[1]):
    dataTotal.append(df.values[0:, i:i + 1])
getPBPE = index_indicator(index[0], start_date=lastTradeDate, end_date=lastTradeDate).iloc[:, 1:3]
for i in range(getPBPE.shape[1]):
    dataTotal.append(getPBPE.values[0:, i:i + 1])
for i in range(dfSZZS.shape[1]):
    dataTotal.append(dfSZZS.values[0:, i:i + 1])
for i in range(dfHS300.shape[1]):
    dataTotal.append(dfHS300.values[0:, i:i + 1])
for i in range(dfSZCZ.shape[1]):
    dataTotal.append(dfSZCZ.values[0:, i:i + 1])

price = 0
for i in range(len(model)):
    # print(model[i])
    # print(dataTotal[i][0][0])
    price = price + model[i] * dataTotal[i][0][0]
# print(data)
# print(getPBPE)
print(price)