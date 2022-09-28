import pandas as pd
from tick import Tick
from tick import corr_vectors, plot_nvector_any, load_vectors
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY, DAILY, WEEKLY
import numpy as np
import csv
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt



vector_correlation = 0.45

v_width = 200
sber = Tick('SBER_220701_220921.csv', v_width=v_width)


#gaz = Tick('GAZP_210101_220829.csv')
#luk = Tick('SNGS_210101_220829.csv')
#vtb = Tick('VTBR_210101_220829.csv')
#nvtk= Tick('NVTK_210101_220829.csv')
#sber.set_interval('20220801', '20220901')

print('sberend', sber.end)

#print("autocor {}".format(sber.data['CLOSE'].pct_change().autocorr()))

#sber.modelling(150)
#sber.enrich(alum, 'AL')
#sber.enrich(usd, 'USD')
#print(sber.data['VAR'])

#sber.promodelling()
#sber.modelling()
#alum.modelling()
print(sber.vol)
print(sber.price)

abc = sber.sumvol()
sber.target_correlation = 0.9

#sber.v_prominence = 1
#sber.v_width = 1
#sber.foreseeing_last(600)


lag_width = 10

last_lag = sber.vectors.shape[0]-lag_width
last_vector = sber.vector(last_lag, lag_width)
sber.load_vectors()
print("vectors", len(sber.vectors))

#sber.create_training_set(10)
#plot_nvector_any(last_vector, rgb=(0.9, 0, 0))

print("sber.vector", sber.vector(last_lag, lag_width))
print('vector-10', sber.vector(last_lag,10))

lag_width = 10

current_tick = Tick('SBER_220701_220921.csv', v_width=100)
#50 is about 1 day for a lag of 20
#400 is about 1 month for a lag of 10
last_lag = current_tick.vectors.shape[0] - 10
last_vector = current_tick.vector(last_lag, 10)
sber.predict_vectors(last_vector, 1)
#reference_vector = current_tick.vector(last_lag, 13)
#plot_nvector_any(reference_vector, rgb=(0.9, 0, 0), divider=10)
plt.show()
#sber.predict_vectors(last_vector, 2)
#sber.predict_vectors(last_vector, 3)
exit()
#sber.search_vector(last_vector, lag_width, vector_correlation)
result_all = []
last_vectors = np.zeros((0,lag_width,3))

pl_n=1
for j in range(50, 100, 50):
    current_tick = Tick('SBER_220701_220921.csv', v_width=j)
    last_lag = current_tick.vectors.shape[0] - lag_width
    last_vector = current_tick.vector(last_lag, lag_width)
    last_vectors = np.concatenate((last_vectors, last_vector[np.newaxis,:,:]))
    print('last', last_vectors)
    print('shape', last_vectors.shape)
    print('checking:', j)
    plt.subplot(4, 2, pl_n)
    pl_n = pl_n + 1
    plot_nvector_any(last_vector, rgb=(0.9, 0, 0))
    plt.subplot(4, 2, pl_n)
    pl_n=pl_n+1

    for i in np.arange(0.3, 0.56, 0.02):
        res = sber.search_vectors(last_vector, lag_width, i)
        print('searching:', i, res)
        if type(res) is int:
            if res == 0:
                print("thats all for", j)
                break
        else:
            print('result_all', result_all)
            result_all.append(res)
            for drawe in res:
                vectorr = sber.vector(int(drawe), lag_width * 3)
                plot_nvector_any(vectorr, divider=lag_width)

        #print("search", i, res)

#for vec in result_all:
#    vectorr = sber.vector(int(vec[0]), lag_width*3)
#    plot_nvector_any(vectorr, divider=lag_width)
#    plt.show()
#for new in last_vectors:
#    plot_nvector_any(new, rgb=(0.9, 0, 0))
plt.show()

print('done vectoring')

#sber.whats_next()



plt.subplot(4, 1, 1)
a = sber.start
print("sberstart")
print(a)
b = sber.end
print("sberend")
print(b)

for m in rrule(MONTHLY, dtstart=a, until=b):
    mm = m + relativedelta(months=+1)
    print(m, mm)
    sber.set_interval(m, mm)
    volu = sber.sumvol()
    plt.plot(volu, label=m)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
sber.set_interval(a, b)



'''plot_nvector_any(last_vector, rgb=(0.9, 0, 0))
for ve in range(0, gaz.vectors.shape[0]-lag_width):
    checkvec = gaz.vector(ve, lag_width)
    doublevec = gaz.vector(ve, lag_width*2)#rg = (sber.vectors.shape[0] - ve)/sber.vectors.shape[0]
    i = corr_vectors(last_vector, checkvec, vector_correlation)
    #i = corr_vectors(sber.vector(last_lag, lag_width), sber.vector(ve, lag_width), vector_correlation)
    if i == 1:
        plot_nvector_any(doublevec)
plt.show()

for m in rrule(freq=WEEKLY, dtstart=a, until=b):
    mm = m + relativedelta(days=+1)
    print(m, mm)
    sber.set_interval(m, mm)
    #volu = sber.sumvol()
    plt.plot(sber.data['CLOSE'].pct_change(), label=m)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
sber.set_interval(a, b)'''

peaks, vectors = sber.minmax()
plt.subplot(4, 1, 2)
plt.plot(peaks, sber.price[peaks],"xr")
plt.plot(sber.price)
for i in range(len(peaks)-1):
    x1, y1 = [peaks[i], peaks[i+1]], [sber.price[peaks[i]], sber.price[peaks[i+1]]]
    plt.plot(x1, y1)
print("vectors")
print(vectors)

plt.subplot(4, 1, 3)
x=0
y=0
for i in range(len(vectors)):
    x1, y1 = [x, x + vectors[i,0]], [y, y + vectors[i,1]]
    xv, yv = [x, x + np.sign(vectors[i,0])*vectors[i,2]], [y, y]
    x = x + vectors[i, 0]
    y = y + vectors[i, 1]
    plt.plot(y1, x1)
    plt.plot(yv, xv)
plt.subplot(4, 1, 4)


plt.plot(sber.price[-5000:])

plt.show()

#mm = sber.turning_points(peaks)
#mm.plot()
#plt.show()






# sber.save4prophet()
#sber1 = np.genfromtxt('SBER.csv', delimiter=';')
#print(sber1)

#with open('SBER.csv', newline='') as csvfile:
#    csvsber = csv.reader(csvfile, delimiter=';')
#    for i in csvsber:
#        print(i)
#    df = pd.DataFrame(csvsber)

#volume = pd.DataFrame()
#dff = pd.read_csv('SBER.csv', delimiter=';')
#ros = pd.read_csv('RSTI.csv', delimiter=',')
#print(ros['TIME'])
#ros['DATETIME'] = ros['DATE'].astype(str) + ros['TIME'].astype(str)
#ros['DATE'] = pd.to_datetime(ros['DATE'], format='%Y%m%d')
#ros['TIME'] = pd.to_datetime(ros['TIME'], format='%H:%M')
#ros['DATETIME'] = ros['DATE'] + ros['TIME']
#ros['DATETIME'] = pd.to_datetime(ros['DATETIME'], format='%Y%m%d%H:%M')
#ros.set_index('DATETIME', inplace=True)
#print(ros['DATETIME'])

#volume['CLOSE'] = dff['CLOSE'].astype(int).unique()
#volume['SUMVOL'] = 0
#volume.set_index('CLOSE', inplace=True)
#print(volume.loc[225,'SUMVOL'])

#print(volume)
#volume.apply(lambda a: [int(b) for b in a])
#volume = pd.to_numeric(volume, downcast='integer')
#print(dff['TICKER'])
#dff['DATE'] = pd.to_datetime(dff['DATE'], format='%Y%m%d')
#dff.set_index('DATE', inplace=True)
#print('index set')
#dff['VOL'] = (dff['VOL'] - dff['VOL'].min()) / (dff['VOL'].max() - dff['VOL'].min())
#dff['CLOSE'] = (dff['CLOSE'] - dff['CLOSE'].min()) / (dff['CLOSE'].max() - dff['CLOSE'].min())
#df1=dff['CLOSE']
#df1['VO1L'] = dff['VOL']
#df1 = dff.loc[:, ('CLOSE', 'VOL')]
#df1['ALLVOL'] = df1['CLOSE']*df1['VOL']
#for i, row in df1.iterrows():
    #print(row[1], int(row[0]))
 #   volume.loc[int(row[0]),'SUMVOL']+=row[1]

#print(volume)



#volume.sort_index(inplace=True)
#plt.plot(volume['SUMVOL'])
#abc = sber.sumvol()
#plt.plot(ros.CLOSE)
