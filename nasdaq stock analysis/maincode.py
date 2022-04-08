import numpy as np
from numpy import *
import pandas as pd
import yfinance as yf
nasdaqdata = pd.read_csv('simplend.csv')
ndsymbol = nasdaqdata['Symbol']
ndsymbollist = []
for d in range(len(ndsymbol)):
    ndsymbollist.append(ndsymbol[d])
ndstabledata = {}
for syb in ndsymbollist:
    data = yf.download(tickers=syb, period='720d', interval='1d')

    data = pd.DataFrame(data)
    # print(data['Open'])
    rd = data.loc[:,['Close']]
    # print(rd)
    # print(rowdata.iloc[0])
    # print(rowdata.stack().min())
    ampdata = []
    for i in range(len(rd)-1):
        amp = (rd.iloc[i+1]-rd.iloc[i])/rd.iloc[i]*100
        ampdata.append(amp)
        # if abs(amp.item())>=8:
            # print('This stok is unstable.amp:{}'.format(amp))
    if mean(ampdata)<=0.001:
        ndstabledata.update({syb:mean(ampdata)})
    else:
        pass
print(ndstabledata)
