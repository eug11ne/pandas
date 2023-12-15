from tick_vectors import Tick_vectors, plot_nvector_any
import pandas as pd
from pathlib import Path

from vectors import Vectors
from tick import Tick
import numpy as np

def main():
    nums = [5]#, 10, 15]
    ema = (300, 15000)
    normalize = True#240, 180, 120, 100, 80, 60]#[30, 60, 120, 180, 240, 480]

    p_train = Path('trainingdata')
    p_test = Path('testdata')
    for i in nums:
        ttv1 = load_save(p_train, i, ema, normalize)
        ttv2 = load_save(p_test, i, ema, normalize)
        print('Total vectors saved:', ttv1+ttv2)

def load_save(filename, length, ema, normalize=True):
    ttv=0

    for file in filename.glob('*.csv'):
        sber = Tick_vectors.from_file(file, add_mean=length, ema=ema, normalize=normalize)
        ttv += len(sber.vectors)
        sber.save_vectors(f"Sber{length}_{ema[0]}_{ema[1]}.vec")


    return ttv

if __name__ == "__main__":
    main()