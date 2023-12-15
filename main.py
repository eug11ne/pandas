import pandas as pd

from tick import Tick, load_vectors
from vectors import Vectors, temp_pred
from tick_vectors import Tick_vectors, plot_nvector_any
from tick_model import Tick_model
from training_set import Tick_training_set
from tick_time import Tick_time, plot_candles
from tick_correlation import Tick_lags

from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY, DAILY, WEEKLY
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
import csv
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
print(date.today().weekday())
factor = 50






#filename='GAZP_big_training_set.csv'
filename='SBER_200101_201231.csv'
#filename='SBER_220101_230526.csv'
#filename = 'NVTK_all_data.csv'
#abc = Tick_vectors.from_file(filename, add_mean=5, ema=(250,1500))
#for i in abc.diff_ema1:
#    print(i)
#print(abc.diff_ema1.max())
#print(abc.diff_ema1.min())
#plt.plot(abc.ema2)
#plt.plot(abc.ema1)
#plt.plot(abc.diff_ema2)
#plt.plot(abc.diff_ema1)

#abc.plot()

#plt.show()
#print(abc.vectors)
#exit()

#lens = [5, 7, 15, 20, 25]
# 20-2, 30-2. 30-3
#filename='Sber5_250_1500.vec'
#filename='Sber10_250_1500.vec'
#filename='Sber15_250_1500.vec'

columns2 = [['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'VOLX', 'VOL', 'TOEMA1', 'TOEMA2', 'EMADIFF', 'EMA1', 'EMA2', 'DIFF1', 'DIFF2', 'DDIFF1', 'DDIFF2'],
['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'VOLX', 'VOL', 'TOEMA1', 'TOEMA2', 'EMADIFF', 'EMA1', 'EMA2', 'DIFF1', 'DIFF2'],
['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'VOLX', 'VOL', 'TOEMA1', 'TOEMA2', 'EMADIFF', 'EMA1', 'EMA2'],
['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'VOLX', 'VOL', 'TOEMA1', 'TOEMA2', 'EMADIFF'],
['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'VOLX', 'VOL', 'EMA1', 'EMA2'],
['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'VOLX', 'VOL'],
['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'VOLX'],
['X', 'TIME', 'RATIO', 'SIN', 'VOLX'],
['X', 'RATIO', 'SIN', 'VOLX'],
['X', 'RATIO', 'SIN', 'VOL'],
['X', 'RATIO', 'SIN']]
notbad = [['X', 'DELTAX', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'VOL', 'VOLX', 'SUM'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'VOL', 'VOLX'],
['X', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF'],
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF'], #not bad at 15
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'DELTADIFF', 'VOLX'],
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],
['X', 'DELTAX', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],# 47 bigs, 5.2, more bigs but less coefficient compared to +LEN
['X', 'DELTAX', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'VOLX'],
['X', 'DELTAX', 'LEN', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOL'] #39 bigs, 5.5

            ]

columns = [['X', 'DELTAX', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],
    ['X', 'DELTAX', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'VOL', 'VOLX', 'SUM'],
    ['X', 'DELTAX', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],
['X', 'DELTAX', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],# 47 bigs, 5.2, more bigs but less coefficient compared to +LEN
['X', 'DELTAX', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'VOLX'],
['X', 'DELTAX', 'LEN', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],
['X', 'SIN', 'RATIO', 'EMADIFF','VOL'],
['X', 'SIN', 'RATIO', 'EMADIFF','VOLX'],
['X', 'DELTAX', 'LEN', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOL', 'DELTADDIFF'],
['X', 'DELTAX', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOLX', 'DELTADDIFF'],
['X', 'DELTAX', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'VOLX'],
['X', 'DELTAX', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],
['X', 'DELTAX', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'VOLX', 'EMA2'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMA1', 'EMA2','VOL'],
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'VOL'],
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'VOLX'],
['X', 'DELTAX', 'LEN', 'TIME', 'SIN', 'RATIO', 'DELTADDIFF', 'VOL'],
['X', 'DELTAX', 'LEN', 'TIME', 'SIN', 'RATIO', 'EMADIFF', 'DELTADDIFF', 'VOL'],
['X', 'DELTAX', 'TIME', 'SIN', 'RATIO', 'EMADIFF','VOL', 'VOLX'],
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF'],
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'DELTADIFF', 'VOLX'],
['X', 'DELTAX', 'SIN', 'RATIO', 'VOLX', 'EMADIFF'],
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'DELTADIFF', 'VOLX'],
['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF', 'DELTADIFF'],
#['X', 'TIME', 'DELTAX', 'LEN', 'RATIO', 'SIN', 'VOLX', 'VOL', 'TOEMA1', 'TOEMA2', 'EMADIFF', 'EMA1', 'EMA2', 'DIFF1', 'DIFF2', 'DDIFF1', 'DDIFF2', 'DELTADIFF', 'DELTADDIFF'],
#['X', 'DELTAX', 'TIME', 'LEN', 'SIN', 'RATIO', 'EMADIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'VOL', 'EMADIFF', 'DELTADIFF', 'DELTADDIFF'], #,'SUM'],#, 'VOL'], 'EMADIFF', 'RATIO', 'VOL', 'VOLX', 'SUM',
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADDIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADDIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADDIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADDIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADDIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADDIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADDIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADIFF', 'DELTADDIFF'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADIFF', 'VOLX'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DELTADIFF', 'VOL'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DIFF1'],
['X', 'DELTAX', 'SIN', 'RATIO', 'EMADIFF', 'DIFF2'],
['X', 'DELTAX', 'SIN'],
['X', 'LEN', 'VOL'],
['X', 'TIME', 'LEN', 'VOL'],
['X', 'EMA1', 'EMA2'],
['X', 'DELTAX', 'EMA1', 'EMA2'],
['X', 'EMA1', 'EMA2', 'EMADIFF'],
['X', 'EMA1', 'EMA2', 'SIN'],
['X', 'EMA1', 'EMA2', 'SIN', 'DELTAX'],
['X', 'EMA1', 'EMA2', 'SIN', 'DELTAX', 'RATIO'],
['X', 'EMA1', 'EMA2', 'SIN', 'DIFF1', 'DIFF2'],
['X', 'EMA1', 'EMA2', 'SIN', 'DELTAX', 'RATIO', 'DIFF1', 'DIFF2']
#['X', 'DELTAX', 'RATIO', 'EMA1', 'EMA2', 'SIN'],
#['X', 'TIME', 'DELTAX', 'RATIO', 'EMA1', 'EMA2', 'SIN'],
#['X', 'TIME', 'EMA1', 'EMA2', 'SIN']
            ]


#filename='Sbergazp_vectors.csv'
lens = [17]#, 20, 20, 20]#, 15, 15]#, 20, 20]#, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]#, 20, 20]#15, 16, 17, 18, 19, 20, 21, 22, 23]
nns = [(4, 4)]#, (5, 6), (5, 6), (5, 6)]#, (4, 3), (4, 3)]#, (3, 3), (3, 3)]#, (3, 4),(3, 4), (3, 4), (3, 4), (3, 4), (3, 4), (3, 4), (3, 4), (3, 4)]#, (8,3), (7,6)]#, (5,3), (5,3), (5,3), (5,3)]
mins = 5
#filename=f'Sber{mins}_250_1500.vec'
filename='Sber5_300_15000.vec'
train=True
cum_pred = []
preds = []
ytest = []
#train=False
model_number=0
if train:
    for i, nn in zip(lens, nns):
        print("i", i)
        trset = Tick_training_set(filename, i, 'simple', steps=3, nn=nn, columns=columns[model_number]) #Sbergazp_vectors.csv Sber_vectors.vec
        #trset = Tick_training_set(filename, i, 'all') #Sbergazp_vectors.csv Sber_vectors.vec
        preds, ytest = trset.create_nn(model_number=model_number, mins=mins, lstm=True)
        cum_pred = cum_pred + preds
        model_number+=1
        #trset.create_model()
else:

    ve1 = Vectors.from_file("SBER_220101_230629.csv", add_mean=70, comb_ratio=0.9)#.slice(0,-1)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
    stb = 1
    temp_pred(ve1, 15, stb, 'simple', nn=True, model_number=2)
    ve1.plot()
    plt.show()

''' cum_pred = [i/(len(lens)) for i in cum_pred]
right_direction=0
wrong_direction=0
good_guess=0
norm_guess=0
for i, j in zip(cum_pred, ytest):#Y_test.to_numpy().T):
    print(f"predicted: {i}, actual: {j}")
    if i*j>0:
        right_direction+=1
    else:
        wrong_direction+=1
    if abs(j-i) < 0.1:
        good_guess+=1
    elif abs(j-i) < 0.2:
        norm_guess+=1
print(f"Good ones: {good_guess}, ok ones: {norm_guess}, right direction guess: {right_direction}, wrong direction guess: {wrong_direction}, coefficient: {right_direction/wrong_direction}")
'''
exit()



ve2 = Vectors.from_file("SBER_220101_230629.csv", add_mean=15)
lens = [20]
preds = []
all_deltas=[]
answers = []
deltas = []

for len in lens:
    preds = []

    avgs = []

    for i in range(50, 74):
        ve1 = ve2.slice(0,i)
        #pred1 = ve1.fast_predict(f"keras_1step_{len}_simple_1.nnn", len, 'simple', nn=True)
        pred1 = ve1.fast_predict(f"keras_1step_{len}_mins15_1.nnn", len, 'simple', nn=True)
        pred6 = ve2.right_answer(i,3)
        delta = pred6 - pred1[0][0]
        deltas.append(f'{len}: {pred1[0][0]}, {pred6}, {delta}')

        avgs.append(delta)
        all_deltas.append(delta)
    deltas.append(f'Mean delta for {len}: {np.abs(np.array(avgs)).mean()}')

for j in deltas:
    print(j)

print('total delta: ', np.abs(np.array(all_deltas)).mean())
exit()









ve = Vectors.from_file("SBER_220101_230526.csv", add_mean=240, comb_ratio=0.9)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#temp_pred(ve, 15, 0)
for p in range(10):
    temp_pred(ve, 3, p)
#temp_pred(ve, 15, 1)
#temp_pred(ve, 15, 2)
#temp_pred(ve, 15, 3)
#temp_pred(ve, 15, 4)
#temp_pred(ve, 15, 5)
#temp_pred(ve, 15, 7)

ve.plot()
plt.show()
a1 = ve.fast_predict("keras_1step_13_simple_plus.nnn", 10, 'simple', nn=True)
a2 = ve.fast_predict("keras_2step_13_simple_plus.nnn", 10, 'simple', nn=True)
a3 = ve.fast_predict("keras_3step_13_simple_plus.nnn", 10, 'simple', nn=True)
print(a1, a2, a3)



exit()

ve1 = Vectors.from_file("SBER_220101_230526.csv", add_mean=500, comb_ratio=0.9).slice(0,-1)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
a1 = ve1.fast_predict("gb_model-1step_12_std.pkl", 9, 'std')
a2 = ve1.fast_predict("gb_model-2step_12_std.pkl", 9, 'std')
a3 = ve1.fast_predict("gb_model-3step_12_std.pkl", 9, 'std')
ve1.add(a1,12400,1,1,1)
ve1.add(a2,12400,1,1,1)
ve1.add(a3,12400,1,1,1)

ve2 = Vectors.from_file("SBER_220101_230526.csv", add_mean=500, comb_ratio=0.9).slice(0,-1)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
a12 = ve2.fast_predict("gb_model-1step_10_std.pkl", 7, 'std')
a22 = ve2.fast_predict("gb_model-2step_10_std.pkl", 7, 'std')
a32 = ve2.fast_predict("gb_model-3step_10_std.pkl", 7, 'std')
ve2.add(a12,12400,1,1,1)
ve2.add(a22,12400,1,1,1)
ve2.add(a32,12400,1,1,1)

print(a1, a2, a3)
ve.plot()
ve1.plot(divider=ve1.length-3)
ve2.plot(divider=ve1.length-3)
plt.show()
exit()

steps = [5, 7, 9, 15, 20]

for i in steps:
    trset = Tick_training_set('Sbergazp_vectors.csv', i, 'std')
    trset.create_model()
exit()

trset = Tick_training_set('Sber_vectors_mins-less.csv', 25, 'std')
trset.create_model()
exit()


factor = 50
#sber-01012015_25082022.txt
abc = Vectors.from_file("SBER_220101_230526.csv", add_mean=30, comb_ratio=0.9)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
abv = Vectors.from_file("SBER_220101_230526.csv", add_mean=60, comb_ratio=0.9)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abv = Vectors.from_file("SBER_220101_230526.csv", v_width=0.02, v_prominence=0.04, add_mean=60, comb_ratio=0.5)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
abg = Vectors.from_file("SBER_220101_230526.csv", add_mean=120, comb_ratio=0.9)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
abf = Vectors.from_file("SBER_220101_230526.csv", add_mean=240, comb_ratio=0.9)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abv = Tick_vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=60, comb_ratio=1, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abg = Tick_vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=120, comb_ratio=1, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abf = Tick_vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=240, comb_ratio=1, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abc.flat_check()


#abf.plot()
abc.plot()
abg.plot()
abv.plot()
abf.plot()
plt.show()

exit()


ve = Vectors.from_file("SBER_220101_230526.csv", v_width=0.005*factor, v_prominence=0.01*factor, add_mean=60, comb_ratio=1)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
a1 = ve.fast_predict("gb_model-1step_28_std.pkl", 25, 'std')
a2 = ve.fast_predict("gb_model-2step_28_std.pkl", 25, 'std')
a3 = ve.fast_predict("gb_model-3step_28_std.pkl", 25, 'std')
ve.add(a1,12400,1,1,1)
ve.add(a2,12400,1,1,1)
ve.add(a3,12400,1,1,1)
print(a1, a2, a3)
ve.plot(divider=ve.length-3)
plt.show()
exit()


#sber-01012015_25082022.txt - all sber
abc = Vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=30, comb_ratio=1)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
abv = Vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=60, comb_ratio=1)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
abg = Vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=120, comb_ratio=1)#, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abv = Tick_vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=60, comb_ratio=1, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abg = Tick_vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=120, comb_ratio=1, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abf = Tick_vectors.from_file("sber-01012015_25082022.txt", v_width=0.005, v_prominence=0.01, add_mean=240, comb_ratio=1, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
#abc.flat_check()


#abf.plot()
abc.plot()
abg.plot()
abv.plot()
plt.show()

exit()

abf = Tick_vectors.from_file("SBER_220101_230526.csv", v_width=0.005, v_prominence=0.01, add_mean=30, comb_ratio=1, flat=True)#, , add_mean=50, comb_ratio=1, add_mean=10comb_ratio=0.9,add_mean=100)#, comb_ratio=0.9, add_mean=200)
ve = Vectors(abf.vectors, 120, abf.peaks)
a1 = ve.fast_predict("gb_model-1step_18_std.pkl", 15, 'std')
a2 = ve.fast_predict("gb_model-2step_18_std.pkl", 15, 'std')
a3 = ve.fast_predict("gb_model-3step_18_std.pkl", 15, 'std')
print(a1, a2, a3)
exit()

trset = Tick_training_set('Sber_vectors_mins-less.csv', 15, 'std')
trset.create_model()
exit()


trset = Tick_training_set('Sber_vectors_flat.csv', 25, 'std')
trset.create_model()
exit()



abc = Vectors.from_file('SBER_220101_221231.csv', 2, 2)
abc.fast_predict('gb_model-1step_13_sum.pkl', 10, 'sum')
#print(abc.vectors)
a = Tick_time("SBER_210101_211231.csv")
for i in range(8, 15):
    a.week(i)
plt.show()
exit()
print(a.find_one(2021, 8, 31, 12))

plot_candles(a.find_range(2021, 8, 23, (10, 19)))
plot_candles(a.find_range(2021, 8, 24, (10, 19)))
plot_candles(a.find_range(2021, 8, 25, (10, 19)))
plot_candles(a.find_range(2021, 8, 26, (10, 19)))
plot_candles(a.find_range(2021, 8, 27, (10, 19)))
plt.show()
#plot_candles(a.find_range(2021, 8, 29, (10, 18)))
#a.find(hours=(10,12))
exit()


''' clf = LogisticRegression(max_iter=1000)
clf.fit(df_train,df_train_target)
joblib.dump(clf, "model.pkl")

во втором:

clf2 = joblib.load("model.pkl")
y_pred = clf2.predict(df_val) '''
vector_correlation = 0.45

v_width = 0.1
v_prominence = v_width*2
trset = Tick_training_set('ALRSSBER_vectors.csv', 50, 'std')
#trset = Tick_training_set('5d-vectors-10k.csv', 100, 'std')
#trset = Tick_training_set('comb-vectors-10k.csv', 20, 'sum2v', features=7)
# trset = Tick_training_set('all-vectors-10k.csv', 50, 'sum', features=3)
trset.create_model()
#sber = Tick_vectors.from_file('SBER_170101_171230.csv', v_width=v_width, v_prominence=v_prominence)
#abc = Vectors.from_file('SBER_180101_181231.csv', 0.5, 1)
#bcd = Vectors.from_df(sber.data, 1, 1)

#print(abc.vectors)
exit()
#sber = Tick_vectors.from_file('SBER_170101_171230.csv', v_width=v_width, v_prominence=v_prominence)
#rv = Vectors(sber.vectors, sber.price[0])
#a = rv.vector_to_predict_sum(10)
#print(a)
#exit()

#gaz = Tick('SBER_220701_220921.csv')



#trset.find_best_model()
exit()

sber = Tick_model('1min.txt', v_width=v_width, v_prominence=v_prominence)
sber.vector_to_predict(10)
sber.create_training_set(10)
lag_width = 10
rv = Vectors(sber.vectors, sber.price[0])
#pred = rv.fast_predict("trs_model-1step_13.pkl", 10))
#exit()
#rv.create_training_set(10)
#rv.vector_to_predict(10)
#rv.vector_to_predict_cos(10)
#exit()
#rv.fast_predict()

rvv = rv.slice(100, 20)
#exit()
pred = rvv.fast_predict('trs_model-1step_13.pkl', 10)
pred2 = rvv.fast_predict('trs_model-2step_13.pkl', 10)
pred3 = rvv.fast_predict('trs_model-3step_13.pkl', 10)
print("pred", pred)
rvv.plot()
rvv.add(pred, 200, 2)
rvv.add(pred2, 200, 2)
rvv.add(pred3, 200, 2)
rvv.plot(divider=rvv.vectors.shape[0]-3)
plt.show()

#sber.create_model_for_vectors(10)
exit()

#sber_c = Tick_lags('SBER_210101_211231.csv')
exit()
#nums = [0.5]#, 1, 2, 3]

#for num in nums:
#    current_tick = Tick_model('1.txt', v_width=num, v_prominence=num*1)
#    current_tick.rich_vectors.slice(current_tick.vectors.shape[0]-10, 10).plot()

#current_tick = Tick_model('1.txt', v_width=3, v_prominence=7)
#print("Pred ", current_tick.fast_predict('model-1step.pkl', 10))
#current_tick.plot()

#current_tick.rich_vectors.slice(5, 10).plot()
#current_tick.rich_vectors.plot()
#plt.show()
#exit()
#sber_c.corr()
#len = 15

#for b in np.arange(0.09, 1, 0.1):
#    results = sber_c.is_one_level(b, len)
#    for r in results:
#        sber_c.plot_lag(r, len*10, title=str(b))
#    plt.show()
#exit()
#gaz = Tick('GAZP_210101_220829.csv')
#luk = Tick('SNGS_210101_220829.csv')
#vtb = Tick('VTBR_210101_220829.csv')
#nvtk= Tick('NVTK_210101_220829.csv')
#sber.set_interval('20220801', '20220901')

print('sberend', sber.end)

#print("autocor {}".format(sber.data['CLOSE'].pct_change().autocorr()))

#sber.enrich(alum, 'AL')
#sber.enrich(usd, 'USD')
#print(sber.data['VAR'])

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
sber.create_model_for_vectors(10)
exit()

current_tick = Tick('1min_10052022_18102022.txt', v_width=200)
print('PREDICTING...')
print("PREDICTION 1", current_tick.fast_predict('model-1step.pkl', 10))
print("PREDICTION 2", current_tick.fast_predict('model-2step.pkl', 10))
print("PREDICTION 3", current_tick.fast_predict('model-3step.pkl', 10))

exit()
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
    current_tick = Tick('Сбербанк_1min_10052022_06102022.txt', v_width=j)
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
