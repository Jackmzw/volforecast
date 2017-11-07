import dbstock as db
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt

halftradinglist = np.loadtxt("./HalfTradingDays.txt",dtype="int64")

days = db.PNHistory('10107')
n = len(days)
Day = np.zeros(n)
for i in range(n):
    Day[i] = int(days[i][0])

ind = (Day>=19960000) & (Day<20170000)
day2016 = Day[ind]
n = len(day2016)

'''
ind = (day>=19960000) & (day<19970000)
day1996 = day[ind]
ind = (day>=20060000) & (day<20070000)
day2006 = day[ind]
ind = (day>=20160000) & (day<20170000)
day2016 = day[ind]
'''
sp = np.loadtxt("./sp1500.txt",dtype="int64")
ss = len(sp)
M = np.zeros((ss,n))
for i in range(ss):
    print(i)
    days = db.PNHistory(str(sp[i,0]))
    for day in days:
        if (int(day[0]) in day2016) and (int(day[0])>=sp[i,1]) and (int(day[0])<=sp[i,2]):
            ii = np.where(day2016==int(day[0]))[0][0]
            data = db.Query(day[0], day[1])
            tsec = np.array(data[0])
            tick = np.array(data[1])
            nonzero = tick.nonzero()
            tsec = tsec[nonzero]
            tick = tick[nonzero]
            if int(day[0]) in halftradinglist:
                ind  = (tsec >= 1800) & (tsec <= 14400)
                tick = tick[ind]
            else:
                ind = (tsec >= 1800) & (tsec <= 25200)
                tick = tick[ind]
            n1 = len(tick)
            if n1 == 0:
                M[i,ii] = 1
            else:
                ret = np.diff(np.log(tick))
                ret = ret[ret.nonzero()]
                nn = len(ret)
                if nn > 12:
                    M[i,ii] = 3
                else:
                    M[i,ii] = 2
np.savetxt("sp1500_M.csv",M,delimiter=",")

count = np.zeros((3,n))
count[0,] = np.sum(M==1,0)
count[1,] = np.sum(M==2,0)
count[2,] = np.sum(M==3,0)
cs = np.sum(count,0)
plt.plot(count[0,],label="0")
plt.plot(count[1,],label="1~12")
plt.plot(count[2,],label=">12")
plt.plot(cs,label="1500")
plt.legend()
plt.title('sp1500_2016')
plt.show()


'''
days = db.PNHistory('10107')
n = len(days)
day = np.zeros(n)
for i in range(n):
    day[i] = int(days[i][0])
f = open("day.txt","w+")
f.write("\n".join(['%d' % v for v in day]))
ind = (day>=19960000) & (day<19970000)
day1996 = day[ind]
f = open("day1996.txt","w+")
f.write("\n".join(['%d' % v for v in day1996]))

ind = (day>=20060000) & (day<20070000)
day2006 = day[ind]
f = open("day2006.txt","w+")
f.write("\n".join(['%d' % v for v in day2006]))

ind = (day>=20160000) & (day<20170000)
day2016 = day[ind]
f = open("day2016.txt","w+")
f.write("\n".join(['%d' % v for v in day2016]))
'''
'''
# Choose price or midpoint
days = [[sys.argv[1], sys.argv[2]]]
result = {}

for day in days:

    # Loading and cleaning Data
    data = db.Query(day[0], day[1])
    #print(len(data[1]))
    tsec = np.array(data[0])
    tick = np.array(data[1])
    mp   = np.array(data[2])

    nonzero = tick.nonzero()
    tsec = tsec[nonzero]
    tick = tick[nonzero]
    mp   = mp[nonzero]

    nonzero = mp.nonzero()
    tsec = tsec[nonzero]
    tick = tick[nonzero]
    mp   = mp[nonzero]

    if int(day[0]) in halftradinglist:
        ind  = (tsec >= 1800) & (tsec <= 14400)
        tsec = tsec[ind]
        tick = tick[ind]
        mp   = mp[ind]
    else:
        ind = (tsec >= 1800) & (tsec <= 25200)
        tsec = tsec[ind]
        tick = tick[ind]
        mp   = mp[ind]

    n = len(tsec)

    if n<=12 :
        result[day[0]] = ()
        f = open(picklefile, "wb")
        pickle.dump(result,f)
        f.close()
        continue

    # calculate qmle from price
    T = (tsec[n-1] - tsec[0]) / 23400 / 252
    ret = np.diff(np.log(tick))
    ret = ret[ret.nonzero()]
    nn = len(ret)

    #f = open("MSFT19960422.txt","w+")
    #f.write("\n".join(['%.10f' % v for v in tick]))
    line = day[0]
    qmle_price = {}
    qmle_mp = {}

    if nn > 12:
        qmle_price = QMLE_MA_CI_Final(ret,T,6)
        z = qmle_price
        for i in range(1) :
            line += " %0.6E %d %0.6E %0.6E" % (z['iv'][i], z['q'][i], z['CI'][i][0], z['CI'][i][1])

    else : line += " 0.0 0 0.0 0.0"

    # Calculate for subprice

    interval = 300
    subp300 = subsample(tsec,tick,interval)
    ret = np.diff(np.log(subp300))
    ret = ret[ret.nonzero()]
    nn = len(ret)
    sub300_price = np.sqrt(np.sum(ret**2)*252)
    line += " %0.6E" % (sub300_price)

    interval = 900
    subp900 = subsample(tsec, tick, interval)
    ret = np.diff(np.log(subp900))
    ret = ret[ret.nonzero()]
    nn = len(ret)
    sub900_price = np.sqrt(np.sum(ret**2)*252)
    line += " %0.6E" % (sub900_price)

    # calculate qmle from mp

    ret = np.diff(np.log(mp))
    ret = ret[ret.nonzero()]
    nn = len(ret)

    if nn > 12:
        qmle_mp = QMLE_MA_CI_Final(ret,T,6)
        z = qmle_mp

        for i in range(1) :
            line += " %0.6E %d %0.6E %0.6E" % (z['iv'][i], z['q'][i], z['CI'][i][0], z['CI'][i][1])

    else : line += " 0.0 0 0.0 0.0"

    # Calculate for submp

    interval = 300
    subp300 = subsample(tsec,mp,interval)
    ret = np.diff(np.log(subp300))
    ret = ret[ret.nonzero()]
    nn = len(ret)
    sub300_mp = np.sqrt(np.sum(ret**2)*252)
    line += " %0.6E" % (sub300_mp)

    interval = 900
    subp900 = subsample(tsec, mp, interval)
    ret = np.diff(np.log(subp900))
    ret = ret[ret.nonzero()]
    nn = len(ret)
    sub900_mp = np.sqrt(np.sum(ret**2)*252)
    line += " %0.6E" % (sub900_mp)

    result[day[0]] = (qmle_price, sub300_price, sub900_price, qmle_mp, sub300_mp, sub900_mp)

print(result)
print(line)
'''
