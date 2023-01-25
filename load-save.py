from tick_vectors import Tick_vectors, plot_nvector_any
from vectors import Vectors
from tick import Tick
import numpy as np

nums = [1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.01]

#a = Tick('SBER_220101_221231.csv')
ttv=0

for i in nums:
    sber = Tick_vectors.from_file('big_training_set.csv', v_width=i, v_prominence=i*2) #big_training_set.csv
    vectors = Vectors(sber.vectors, 100)
    vectors.comb(1)
    sber.vectors = vectors.vectors
    #sber = Tick_vectors('sber-01012015_25082022.txt', v_width=i, v_prominence=i*2)
    print(i)
    ttv+=len(sber.vectors)
    sber.save_vectors("comb-vectors-10k.csv")
print('Total vectors saved:', ttv)