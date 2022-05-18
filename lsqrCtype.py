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
import pickle

from scipy import stats
import scipy
import json
import ujson
#colorama.init()
#f = open("output2.txt",'w')
from os import walk
from multiprocessing import Pool, cpu_count


from sklearn.linear_model import LinearRegression
import functools
from functools import lru_cache, cache
from numpy_lru_cache_decorator import np_cache
from numpy.linalg import matrix_power

    

avgResults2 = {}
#@np_cache()
# def correlationCoefficient(X, Y, n) :
#     sum_X = 0
#     sum_Y = 0
#     sum_XY = 0
#     squareSum_X = 0
#     squareSum_Y = 0
      
      
#     i = 0
#     while i < n :
#         # sum of elements of array X.
#         sum_X = sum_X + X[i]
          
#         # sum of elements of array Y.
#         sum_Y = sum_Y + Y[i]
          
#         # sum of X[i] * Y[i].
#         sum_XY = sum_XY + X[i] * Y[i]
          
#         # sum of square of array elements.
#         squareSum_X = squareSum_X + X[i] * X[i]
#         squareSum_Y = squareSum_Y + Y[i] * Y[i]
          
#         i = i + 1
       
#     # use formula for calculating correlation 
#     # coefficient.
#     corr = (float)(n * sum_XY - sum_X * sum_Y)/
#            (float)(math.sqrt((n * squareSum_X - 
#            sum_X * sum_X)* (n * squareSum_Y - 
#            sum_Y * sum_Y)))
#     return corr



def linregress(x, y=None, alternative='two-sided'):
    TINY = 1.0e-20
    if y is None:  # x is a (2, N) or (N, 2) shaped array_like
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            raise ValueError("If only `x` is given as input, it has to "
                             "be of shape (2, N) or (N, 2); provided shape "
                             f"was {x.shape}.")
    else:
        x = np.asarray(x)
        y = np.asarray(y)

    if x.size == 0 or y.size == 0:
        raise ValueError("Inputs must not be empty.")

    if np.amax(x) == np.amin(x) and len(x) > 1:
        raise ValueError("Cannot calculate a linear regression "
                         "if all x values are identical")

    n = len(x)
    xmean = np.mean(x, None)
    ymean = np.mean(y, None)

    # Average sums of square differences from the mean
    ssxm = np.subtract(x,xmean)
    #pprint(x)
    #time.sleep()
    ssxm = matrix_power(x, 2)
    ssxm = np.mean(x, None)
    ssym = np.subtract(y,ymean)
    ssym = nmatrix_power(y, 2)
    ssym = np.mean(y, None)
    #ssxym = mean( (x-xmean * (y-ymean)) )
    print(ssxm, ssym)
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=1).flat
    print(ssxm, ssym)

    time.sleep(5)
    # R-value
    #   r = ssxym / sqrt( ssxm * ssym )
    if ssxm == 0.0 or ssym == 0.0:
        # If the denominator was going to be 0
        r = 0.0
    else:
        r = ssxym / np.sqrt(ssxm * ssym)
        # Test for numerical error propagation (make sure -1 < r < 1)
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0

    slope = ssxym / ssxm
    intercept = ymean - slope*xmean
    if n == 2:
        # handle case when only two points are passed in
        if y[0] == y[1]:
            prob = 1.0
        else:
            prob = 0.0
        slope_stderr = 0.0
        intercept_stderr = 0.0
    else:
        df = n - 2  # Number of degrees of freedom
        # n-2 degrees of freedom because 2 has been used up
        # to estimate the mean and standard deviation
        t = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
        t, prob = scipy.stats._stats_py._ttest_finish(df, t, alternative)

        slope_stderr = np.sqrt((1 - r**2) * ssym / ssxm / df)

        # Also calculate the standard error of the intercept
        # The following relationship is used:
        #   ssxm = mean( (x-mean(x))^2 )
        #        = ssx - sx*sx
        #        = mean( x^2 ) - mean(x)^2
        intercept_stderr = slope_stderr * np.sqrt(ssxm + xmean**2)

    return [slope, intercept, r,
                            prob, slope_stderr,
                            intercept_stderr]



def getCorrelation2(xi, yi):
    avgResults2 = {}
    with open('output.json', 'r') as f:
        avgResults2 = ujson.load(f)
    
    #Constants
    Woft = 500    # width of test -  how many coefficients

    t0 = time.time()
    xi_list = []
    yi_list = []

    symbol = xi.replace('_USDT-1m.json','-')+yi.replace('_USDT-1m.json','')

    print(xi, yi)
    print('downloading data...')

    with open(os.getcwd()+'/coins_text/'+xi, 'r') as f:
        xi_list = ujson.load(f)
        #xi_list = np.load(f)

    with open(os.getcwd()+'/coins_text/'+yi, 'r') as f:
        #xi_list = np.load(f)
        yi_list = ujson.load(f)

    t1 = time.time() 
    temp_xi_np = []
    temp_yi_np = []
    #pprint(xi_list)
   
    t0 = time.time()

    for x in range(len(xi_list)):
        temp_xi_np.append(xi_list[x][0])
        
    for y in range(len(yi_list)):
        temp_yi_np.append(yi_list[y][0])

    # for x in range(len(xi_list)):
    #     if xi_list[x][0] not in yi_list:
    #         del xi_list[x]
    #xi_np =  np.array(xi_list)
    #yi_np =  np.array(yi_list)
    print(xi_list[-1][0], yi_list[-1][0])
    count = 0
    while xi_list[0][0] > yi_list[count][0]:
        #print(xi_list[0][0], yi_list[0][0])
        count+=1
    del yi_list[:count]
    count=0
    while xi_list[count][0] < yi_list[0][0]:
        count+=1
    del xi_list[:count]

    count=1
    while xi_list[-1][0] < yi_list[-count][0]:
        #print(xi_list[0][0], yi_list[0][0])
        count+=1
    if count > 1:
        count-=1
        del yi_list[-count:]
    count=1
    while xi_list[-count][0] > yi_list[-1][0]:
        count+=1
    if count > 1:
        count-=1
        del xi_list[-count:]

    #
        # while xi_np[-1][0] < yi_np[-1][0]:
    #     del xi_np[-1]
    # while xi_np[-1][0] > yi_np[-1][0]:
    #     del yi_np[-1]

    if len(xi_list) != len(yi_list):
        print (xi_list[-1][0], yi_list[-1][0])
        print (xi_list[0][0], yi_list[0][0])
        # if symbol not in avgResults2:
        #     avgResults2[symbol] = {}
        # avgResults2[symbol].append(['Not equal len'])
        # with open('output.json', 'w') as f:
        #     json.dump(avgResults2, f)
        return 0

    t1 = time.time()

    print('done downloading data...', t1-t0)
    #time.sleep(100)
    candle = 500
    #---------


    # if xi_list[0][0] < yi_list[0][0]: 
    #     print('hi1')
    #     newind = 0
    #     ind = 0

    #     for y in xi_list:
    #         if ind == len(yi_list)-1:
    #             break
    #         if y[0] == yi_list[0][0]:
    #             newind = ind
    #             break
    #         ind+=1
    #     print(newind)
    #     ind = 0

    #     del xi_list[:newind]
    
    #     for x in range(len(xi_list)):
    #         temp_xi_np.append(xi_list[x][0])
            
    #     for y in range(len(yi_list)):
    #         temp_yi_np.append(yi_list[y][0])

    #     xi_np =  np.array(temp_xi_np)
    #     yi_np =  np.array(temp_yi_np)

    #     temp_xi_np = []
    #     temp_yi_np = []
    # elif xi_list[0][0] > yi_list[0][0]: 
    #     print('hi2')
    #     newind = 0
    #     ind = 0

    #     for y in yi_list:
    #         if ind == len(xi_list)-1:
    #             break
    #         if y[0] == xi_list[0][0]:
    #             newind = ind
    #             break
    #         ind+=1
    #     print(newind)
    #     ind = 0

    #     del yi_list[:newind]
    
    #     for x in range(len(xi_list)):
    #         temp_xi_np.append(xi_list[x][0])
            
    #     for y in range(len(yi_list)):
    #         temp_yi_np.append(yi_list[y][0])

    #     xi_np =  np.array(temp_xi_np)
    #     yi_np =  np.array(temp_yi_np)

    #     temp_xi_np = []
    #     temp_yi_np = []
    # else:
    #     print('hi')
    #     for x in range(len(xi_list)):
    #         temp_xi_np.append(xi_list[x][0])
            
    #     for y in range(len(yi_list)):
    #         temp_yi_np.append(yi_list[y][0])

    #     xi_np =  np.array(temp_xi_np)
    #     yi_np =  np.array(temp_yi_np)

    #     temp_xi_np = []
    #     temp_yi_np = []
    
    # if xi_list[-1][0] < yi_list[-1][0]:
    #     ind = 0
    #     for y in reversed(yi_list):
    #         #print(ind, len(xi_list))
    #         # if ind == len(xi_list)-1:
    #         #     break
    #         if y[0] == xi_list[-1][0]:
    #             newind = ind
    #             break
    #             #del  xi_list[ind]
    #         ind+=1
    #     xi_list = xi_list[0:-newind]
    #     #np.delete(arr, 1, 0)

    # elif xi_list[-1][0] > yi_list[-1][0]:
    #     ind = 0
    #     for y in reversed(xi_list):
    #         #print(ind, len(xi_list))
    #         # if ind == len(xi_list)-1:
    #         #     break
    #         if y[0] == yi_list[-1][0]:
    #             newind = ind
    #             break
    #             #del  xi_list[ind]
    #         ind+=1
    #     yi_list = yi_list[0:-newind]
    
    # if len(xi_list)%candle != 0:
    #     xi_list = xi_list[len(xi_list)%candle:]
    # if len(yi_list)%candle != 0:
    #     yi_list = yi_list[len(yi_list)%candle:]

    #print((len(xi_list)/500)-1)
    temp = []
    temp2 = []
    temp3 = np.array(None)
    temp4 = np.array(None)

    for x in xi_list:
        temp.append(x[4])
    temp3 = np.array(temp)
    for x in yi_list:
        temp2.append(x[4])
    temp4 = np.array(temp2)

    avgs = []
    avgslst = 0
    lsts = []
    results = []

    #t1 = time.time() 

    #pprint(xi_list)
    #print('done editing array...', t1-t0)
    t0 = time.time()
    # pprint(lsts)
    currentPrice = 0
    if len(xi_list) != len(yi_list):
        print (xi_list[-1][0], yi_list[-1][0])
        print (xi_list[0][0], yi_list[0][0])
        if symbol not in avgResults2:
            avgResults2[symbol] = {}
        avgResults2[symbol].append(['Not equal len'])
        with open('output.json', 'w') as f:
            json.dump(avgResults2, f)
        return 0
        #time.sleep(1000)

    for x in range(int((len(temp)-500))):
        #
        temp5 = np.array(temp3[x:x+(candle-1)])
        temp6 = np.array(temp4[x:x+(candle-1)])
        average = 0
        #pprint(temp3)
        #average = np.corrcoef(temp5, temp6)
        
        avglst = scipy.stats.linregress(temp5, temp6)[0:3]
        # import timeit
        # #m = np.array()
        # m = np.asarray(temp5)
        
        # t0 = time.time()
        # for i in range(1000000):
        #     m= np.mean(temp5, None)
        #     HI = m
        # t1 = time.time()
        # print('done downloading data...', t1-t0)
        #timeit.timeit(np.sum(temp5, temp6), number=10000)
        #print(avglst[0])
        # #avglist = avglst.coef_

        # lsts.append(avglst)

        perc = 0.001
        # #t1 = time.time()
        # xSum = np.sum(temp5)
        # ySum = np.sum(temp6)
        # zSum = np.sum(temp5*temp6)

        # m = (len(temp5) * np.sum(temp5*temp6) - xSum * ySum) / (len(temp5)*np.sum(temp5*temp5) - xSum ** 2) 
        # avgs.append
        # avgs.append(   (xSum - m *ySum) / len(temp5) )
        # avgs.append(average[0][1])

        #model = LinearRegression()
        #model.fit(temp5.reshape(1, -1), temp6.reshape(1, -1))

        #pprint(xi_list)
        #print('done lreg ...', t1-t0)
        horus = 0
        stop = 0
        target = 0
        resul = []
        currentPriceX = temp3[x+candle]
        currentPriceY = temp4[x+candle]
        predPrice = avglst[0]*currentPriceX+avglst[1]
        priceHigher = False
        stop = currentPriceY - (perc * currentPriceY)
        target  = currentPriceY + (perc * currentPriceY)
        if currentPriceY < predPrice:
                priceHigher = True 
        resolved = False

        priceDiff = round(float(currentPriceY - predPrice)/predPrice, 3)

        for y in range(300):
            
            if x+horus == len(yi_list)-1:
                break
            currentPriceY = temp4[x+horus]
            #print(predPrice, temp4[x+horus])
            # time.sleep(1)
            
            # print(stop, currentPrice, target)
            # time.sleep(0.1)
            horus+=1
            
            if currentPriceY >= target:
                temp = [avglst[-1], 1, priceHigher, x, horus, predPrice,priceDiff]
                resul.append(temp)
                resolved = True
                #print(x, horus)#(stop, currentPriceY, target, predPrice)
                # time.sleep(0.1)
                break
            if currentPriceY <= stop:
                temp = [avglst[-1], 0, priceHigher, x, horus, predPrice, priceDiff]
                resolved = True
                resul.append(temp)
                #print(x, horus)#
                #print(stop, currentPriceY, target, predPrice)
                # time.sleep(0.1)
                break
        if not resolved:
            temp = [avglst[-1], 2, priceHigher, x, horus, predPrice, priceDiff]
            resul.append(temp)

        results.append(resul)
        #print(average, avglst[2])

    avgResults = {}
    for res in results[:-1]:
        #print(res[0])
        if not res:
            print('hi')
            time.sleep(100)
        keys = str(round(res[0][0], 1))
        if keys == '-0.0':
            keys = '0.0'
        if keys in avgResults:
            avgResults[keys].append(res[0])
        else:
            avgResults[keys] = [res[0]]

    #pprint(avgResults)
    for x in avgResults:
        totTargetsLow = 0
        totStopsLow   = 0
        totTargetsHigh = 0
        totStopsHigh   = 0
        totErrors = 0
        avgDiff = 0
        count  = 0
        changeLow = 0
        changehigh = 0
        for y in avgResults[x]:
            if not y[2]:
                if y[1] == 1:
                    totTargetsLow += 1
                if y[1] == 0:
                    totStopsLow += 1
                if y[1] == 2:
                    totErrors += 1
            else:
                if y[1] == 1:
                    totTargetsHigh += 1
                if y[1] == 0:
                    totStopsHigh += 1
                if y[1] == 2:
                    totErrors += 1

            avgDiff+=float(y[6])
            count+=1
        avgDiff/=count
        if symbol not in avgResults2:
            avgResults2[symbol] = {}
        if totTargetsLow and totStopsLow and totStopsHigh and totStopsLow:
            changeLow = round(float(totTargetsLow - totStopsLow)/totStopsLow, 3)
            changehigh = round(float(totTargetsHigh - totStopsHigh)/totStopsHigh, 3)
        if x in avgResults2[symbol]:
            avgResults2[symbol][x].append([totTargetsLow, totStopsLow, changeLow, 0, totTargetsHigh, totStopsHigh, changehigh, totErrors, avgDiff])
        else:
            avgResults2[symbol][x] = [[totTargetsLow, totStopsLow, changeLow, 0, totTargetsHigh, totStopsHigh, changehigh, totErrors,avgDiff]]
   # pprint(avgResults2, width=120)
    with open('output.json', 'w') as f:
        json.dump(avgResults2, f)

    xi_list = []
    yi_list = []
    temp_xi_np = []
    temp_yi_np = []
    # clearing temp vars from mem
    t1 = time.time()

    print('done downloading data...', t1-t0)


f = []
for (dirpath, dirnames, filenames) in walk(os. getcwd()+'/coins_text'):
    f.extend(filenames)
    break

#pprint(f)

canidates = []

for xx in f:
    for yy in reversed(f):
        if '.json' in xx and '.json' in yy:
            canidates.append([xx, yy])

            
            
#getCorrelation2( xx,yy)

def t():
    # Make a dummy dictionary
    pool = Pool(processes=(cpu_count() - 2))
    #pool = Pool(1)

    for cans in canidates:
        #getCorrelation2(cans[0], cans[1])
        pool.apply_async(getCorrelation2, args=(cans[0], cans[1]))
    #results = pool.map(getCorrelation2, n)

    pool.close()
    pool.join()


if __name__ == '__main__':
    t()