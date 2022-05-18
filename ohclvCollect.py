import ccxt
import operator
import time
from datetime import datetime
#import numpy as np
from decimal import *
from pprint import pprint
import sys
#from line_profiler import LineProfiler
from termcolor import colored
import colorama
import os
from talib import abstract
from talib.abstract import *
import numpy as np
import math


import json
colorama.init()
f = open("output2.txt",'w')



exchange = ccxt.gateio(
    {
        'apiKey': '',
        'secret': '',
        'enableRateLimit': True,
        'verbose': False,
        'timeout': 30000,
	    #'password': 'JohnBond19',
        'options': {
            'defaultType': 'swap',
        },
    }
 )
exchange1 = ccxt.ascendex(
    {
        'enableRateLimit': False,
        'verbose': False,
        'timeout': 30000,
	    #'password': 'JohnBond19',
        'options': {
            'createMarketBuyOrderRequiresPrice': False
        },
    }
 )
exchange2 = ccxt.kucoin(
    {
        #'enableRateLimit': True,
        'verbose': False,
        'timeout': 30000,
	    'password': 'johnbond19',
        'options': {
            'createMarketBuyOrderRequiresPrice': False
        },
    }
 )
exchange1.apiKey = '' #'603b672562c25d0007a52ecd'  #kucoin
exchange1.secret = ''  #'74608347-46ac-4c96-a3c5-f683410bdb4a'

exchange2.apiKey =  ''  #kucoin
exchange2.secret = ''

utcDone3 = 0
ranOnce = True

#tickers = collectOHCLV(symbol)

#exchange1.checkRequiredCredentials()

average = 0 
#balance = exchange2.fetch_order_book(symbol, limit=100, )
#pprint(balance)

symbols = ['SUSHI/USDT',
    'IOTA/USDT',
    'WAVES/USDT',
    'ADA/USDT',
    'LIT/USDT',
    'XTZ/USDT',
    'BNB/USDT',
    'AKRO/USDT',
    'HNT/USDT',
    'ETC/USDT',
    'XMR/USDT',
    'YFI/USDT',
    'ETH/USDT',
    'ALICE/USDT',
    'ALPHA/USDT',
    'SFP/USDT',
    'REEF/USDT',
    'BAT/USDT',
    'DOGE/USDT',
    'RLC/USDT',
    'TRX/USDT',
    'STORJ/USDT',
    'SNX/USDT',
    'XLM/USDT',
    'NEO/USDT',
    'UNFI/USDT',
    'SAND/USDT',
    'DASH/USDT',
    'KAVA/USDT',
    'RUNE/USDT',
    ]

def checkRSI(symb):
    candleSize = 500
    RSI = abstract.RSI


    candEnd = []
    candleSize = 500

    y = 0
    
    pprint(tickers[0][4])
    pprint(tickers[candleSize-1][4])
    #time.sleep(10)


    tickers[-1][4] = exchange1.fetch_ticker(symbol)['last']
    for x in range(candleSize):
        y = y +1
        #temp1 = x[4]
        #print(candleData['ANKR/USDT'][x][4])
        #if y < 400:
            #   print(x[4])
        candEnd.append(tickers[x][4])

    
    pprint(candEnd)
    clo1 = np.array(candEnd)
    #pprint(tickers)
    #stochRsiK, stochRsiD = STOCHRSI(clo1, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    output1 = RSI(clo1, timeperiod=14)
    print('RSI: ', symbol, ' : ', output1[-1])
    return output1[-1]

candleSize = 200
symbb = 'BTC/USDT'
temp = []
btc = exchange1.fetchOHLCV(symbb, '1h', limit=200)
for x in range(candleSize):
    temp.append(btc[x][4])
    
btcnp = np.array(temp)
    


def getCorrelation(symbb):
    candleSize = 200

    temp = []
    test = exchange1.fetchOHLCV(symbb, '1h', limit=200)
    for x in range(candleSize):
        temp.append(test[x][4])
        
    tester = np.array(temp)
    average = np.corrcoef(btcnp,tester)
    print(symbb, average[0][1])

def getCorrelation2(ohc, temp3, symbb, base):
    avges = {}
    btcnp2 = np.array(temp3)


    temp = []
    test = ohc[symbb]
    for x in range(candleSize):
        temp.append(test[x][4])
        
    tester = np.array(temp)
    average = np.corrcoef(btcnp2,tester)
    sympr = symbb.replace('/USDT', '')
    base = base.replace('/USDT', '')
    #print(base+'/'+sympr, average[0][1])
    avges[base+'/'+sympr] = average[0][1]
    return base+'/'+sympr, average[0][1]

exchange1.load_markets()
print(exchange1.checkRequiredCredentials())

symbols1 = exchange1.symbols

wlist = ['BTC', 'ETH', ':USDT', 'ARTH/USDT', 'ATS/USDT', 'BHD/USDT', 'BXA/USDT', 'COVA/USDT', 'DEEP/USDT', "DEF/USDT", 'DMG/USDT', 'DUO/USDT', 'ERD/USDT', 'FNX/USDT', 'FSN/USDT', 'LAMB/USDT', 'LAMBS/USDT', 'LBA/USDT', 'MHC/USDT','MHUNT/USDT', 'OKB/USDT', 'PAMP/USDT', 'TRY/USDT', 'VALOR/USDT', 'XNS/USDT', 'YAP/USDT']

for f in range(5):
    ind = 0
    for y in symbols1:
        for z in wlist:
            if z in y:
                print('hi1,', symbols1[ind])
                del symbols1[ind]
                print('hi2,', symbols1[ind])
        ind+=1

def getohclv():
    ohclv = {}
    for x in symbols1:
        #print('collecting:', x)
        ohclv[x] = exchange1.fetchOHLCV(x, '1h', limit=200)
    return ohclv

#ohclvdata = getohclv()
#pprint(ohclvdata)    



for y in symbols1:
    candleSize = 200
    btc2 = ohclvdata[y]
    temp2 = []
    for x in range(candleSize):
        temp2.append(btc2[x][4])
    averages = {}

    avgDict = {}
    for x in symbols1:
        #print(x)
        symm, bb = getCorrelation2(ohclvdata, temp2, x, y)
        avgDict[symm] = bb
        

        #exchange1.fetch_ticker(x)
        #time.sleep(0.01)
    avgDict = dict(sorted(avgDict.items(), key=operator.itemgetter(1)))

    counter = 0

    for x in reversed(list(avgDict.items())):
        if counter == 5:
            break
        print(x)
        counter+=1   

    #pprint(avgDict,sort_dicts=False)
