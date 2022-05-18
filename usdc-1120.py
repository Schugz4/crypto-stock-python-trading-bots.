import ccxt
import time
from datetime import datetime
#import numpy as np
from decimal import *
from pprint import pprint
import sys
#from line_profiler import LineProfiler
from termcolor import colored
import colorama
colorama.init()
f = open("output2.txt",'w')

exchange = ccxt.ascendex(
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


#NOTE: exchange api information is required

# exchange.apiKey = '' #'603b672562c25d0007a52ecd'  #kucoin
# exchange.secret = ''  #'74608347-46ac-4c96-a3c5-f683410bdb4a'
exchange.apiKey = '' #''  #kucoin
exchange.secret = ''  #''

exchange.load_markets()
print('CCXT Version:', ccxt.__version__)
symbol = 'BTC/USDT'

balance = exchange.fetchBalance()

pprint(balance)

orderbook = exchange.fetch_order_book (symbol)

#pprint(orderbook['asks'])

base = 0.0001
amountFirstOrder = 5


symbol = 'USDC/USDT'
amount = 20 # amount in XDAI or USDT??



def bidOnly():
    runOnce = False
    runTC = False
    timedCancel = False
    main_order = ['id']
    lastPrice = 0
    while True:
        try:
            orderbook = exchange.fetch_order_book (symbol)
        except:
            print('error')
            time.sleep(1)
            continue
        
        counter = 0
        mainPrice = 1
        total = 0
        goal = 9000
        for x in orderbook['bids']:
            counter = counter + 1 
            total = total + x[1] 
            if lastPrice <= x[0]:
                if total > goal:
                    print('hi')
                    mainPrice = x[0]  # swapa
                    break
                else:
                    print(x[0], ' counter:', counter, ' total: ' + str(total))
            else:
                if total > goal+amount: #or x[0] < goal:
                    print('hi')
                    mainPrice = x[0]  # swap
                    break
                else:
                    print(x[0], ' counter:', counter, ' total: ' + str(total))

        counter = 0
        total2 = 0
        spre = 0.0002
        while orderbook['bids'][counter][0] >= (orderbook['bids'][0][0]-(orderbook['bids'][0][0] * spre)):
            total2 = total2 + orderbook['bids'][counter][1]
            counter = counter + 1

        lowestPrice = orderbook['bids'][0][0]-(orderbook['bids'][0][0] * spre)
   
        lowestPrice = (round(lowestPrice, 5))

        print(colored(('spread res:'+ str(lowestPrice)+ ' total:' + str(total2)), 'yellow'))
    
        utc  = datetime.utcnow().second
        if utc % 5 == 0 and runTC:
            timedCancel = True
            runTC = False
        elif utc % 5 != 0:
            runTC = True

        if lastPrice != mainPrice or timedCancel:
            params = {
                #'timeInForce': 'FOK',  
            }
            timedCancel = False
            if runOnce:
                ci = exchange.cancel_order (main_order['id'], symbol)
                print(colored(ci, 'red'))
                if mainPrice < lowestPrice:
                    time.sleep(10)
                    break
            try:
                main_order = exchange.create_limit_buy_order(symbol, amount, mainPrice, params)   # rememver to change offset
            except:
                balance = exchange.fetchBalance()
                
                stripped = symbol.split('/', 1)[0]
                pprint(balance['total'][stripped])


                if runOnce and balance['total'][stripped] <= amount:
                    ci = exchange.cancel_order (main_order['id'], symbol)
                    print(colored(ci, 'red'))

                    time.sleep(1)

                    print('balance', balance['total'][stripped])
                    orderFill = exchange.fetchOrder(main_order['id'], symbol)
                    fil = orderFill['filled']
                    print("filled:  ", fil)
                    if fil > 0:

                        temp = 0
                        cancelled = main_order['id']
                        main_order['id'] = None
                        filPrice = lastPrice + (lastPrice*0.0002)
                        while temp < fil:

                            buyNow = exchange.create_limit_sell_order(symbol, fil, filPrice, params)
                            temp = buyNow['filled']
                            #if(buyNow['filled'] == fil):
                            #    bought =
                            time.sleep(300)
                            checkBuy = exchange.fetchOrder(buyNow['id'], symbol)
                            print(colored(buyNow, 'green'))
                            print('fill: ', fil, 'temp: ', temp)
                            temp = checkBuy['filled']
                            if buyNow:
                                if temp == fil:
                                    print('Order is (hopefully)  filled!')
                                    break
                                else:
                                    c = exchange.cancel_order (buyNow['id'], symbol)
                                if temp > 0:
                                    fil = fil - temp                  
                                
                            print (colored(c, 'yellow'))
                            checkBuy = None
                        time.sleep(10)
            print(colored(main_order, 'green'))
            runOnce = True
            print(colored(main_order, 'green'))
            runOnce = True
        time.sleep(0.01)
        lastPrice = mainPrice


#def askOnly():



#bidSide()
while True:
    try:
        time.sleep(1)
        bidOnly()
    except Exception as e:
        print(e)
    
#askOnly()
