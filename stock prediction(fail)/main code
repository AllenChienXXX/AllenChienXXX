import numpy
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
data = yf.download(tickers='GOOG', period='2000d', interval='1d')
data = pd.DataFrame(data,index=None)
# my_timestamps = data.index
# my_dates = [ts.strftime("%Y-%m-%d") for ts in my_timestamps]#delete 2021and '-' in the date,turns out I don't really need to do this
# x = my_dates
y = list(data['Close'])#y is the number of close in thin period
amplist = []
for i in range(len(y)-1): #calculate the amplitude in y
    amp = (int(y[i+1])-int(y[i]))/int(y[i])
    amp = amplist.append(amp)
comparativedata = []
for x in range(len(amplist)-2): #divide the datas of amplitude to 5
    callist = amplist[x:x+3]
    comparativedata.append(callist)
    # print(callist)
# print(comparativedata)
#find out does the sequal happens serveral times if no,move the decimal number forward.
# comparativedata = np.array(comparativedata)
# print(np.array(comparativedata))
# print(comparativedata)
tempdata = comparativedata.copy()
# print(tempdata)
decimalnum = 17
# testdata = [[1,2,3,4,5],[1,2,3,4,5],[2,3,5,8,7],[7,2,4,5,6]]
# temptestdata = testdata.copy()
# print(np.around(comparativedata,decimals=5))
while decimalnum > 1:
    flag = 0
    for i in range(len(comparativedata)):#3
        # print(comparativedata[i])
        for x in range(len(tempdata)):
            if comparativedata[i] == tempdata[x]:
                flag +=1
            elif flag > 1:
                print(tempdata[x],flag)
                flag = 0
            else:
                flag = 0
                pass
            # print(comparativedata[i])
            # print(repeatdata)
        
    comparativedata = np.around(comparativedata,decimals=decimalnum)
    comparativedata = comparativedata.tolist()
    tempdata = np.around(tempdata,decimals=decimalnum)
    tempdata = tempdata.tolist()
    decimalnum -= 1
    # print(comparativedata)
if decimalnum <= 1:
    print('Cannot predict')
# plt.scatter(x, y)
# plt.show()
