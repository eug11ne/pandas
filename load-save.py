from tick_vectors import Tick_vectors, plot_nvector_any
from vectors import Vectors
from tick import Tick
import numpy as np

nums = [30, 60, 120, 180, 240, 480]

#a = Tick('SBER_220101_221231.csv')
ttv=0

for i in nums:
    sber = Tick_vectors.from_file('sber-01012015_25082022.txt', v_width=0.005, v_prominence=0.01, add_mean=i, comb_ratio=1, flat=True)#comb_ratio=1) #big_training_set.csv
    vectors = Vectors(sber.vectors, sber.initial_price, sber.peaks)
    #vectors.comb(1)
    sber.vectors = vectors.vectors
    #sber = Tick_vectors('sber-01012015_25082022.txt', v_width=i, v_prominence=i*2)
    print(i)
    ttv+=len(sber.vectors)
    sber.save_vectors("Sber_vectors_mins-less.csv")
print('Total vectors saved:', ttv)