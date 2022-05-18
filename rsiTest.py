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


import json
colorama.init()
f = open("output2.txt",'w')

rsin = 0

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
        #'enableRateLimit': True,
        'verbose': False,
        'timeout': 30000,
	    #'password': 'JohnBond19',
        'options': {
            'createMarketBuyOrderRequiresPrice': False
        },
    }
 )


exchange1.apiKey = '' #
exchange1.secret = ''  #
utcDone3 = 0

def collectOHCLV(symb):
    return exchange1.fetchOHLCV(symb, '1m', limit=500)

symbol = 'ETH/USDT:USDT'
ranOnce = True

tickers = collectOHCLV(symbol)

def checkRSI(symb):
    candleSize = 500
    RSI = abstract.RSI


    candEnd = []
    candleSize = 500

    #tickers = exchange1.fetchOHLCV(symb, '1m', limit=500)

    #pprint(tickers)
    # print(ticks[35], candleData[35][-1])
    #pprint(candleData['ANKR/USDT'])
    #temp2 = tickerData[symb]['close']
    #print(temp2)
    #candEnd[-1] = temp2
    #print()
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










while True:

    # temp = exchange1.fetch_ticker(symbol)['last']
    # pprint(temp)
    # time.sleep(1)
 
    utc  = datetime.utcnow().second
    if utc % 10==0 and utc!=ranOnce:
        rsin = checkRSI(symbol)
        ranOnce = utc
        if float(rsin) > 85 or float(rsin) < 20:
            #ci = exchange.cancel_order (main_order['id'], symbol)
            #print(colored(ci, 'red'))
            
            print('RSI: ', symbol, ' : ', rsin)
            time.sleep(5 * 60)