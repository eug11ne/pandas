from tick import Tick
from tick import corr_vectors, plot_nvector_any, load_vectors
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY, DAILY, WEEKLY
import numpy as np
import csv
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt


for i in np.arange(10, 60, 5):
    sber = Tick('sber-01012015_25082022.txt', v_width=i)
    print(i)
    sber.save_vectors()