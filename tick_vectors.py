import random
import numpy as np
from tick import Tick
import pandas as pd
import math
#from vectors import Vectors
import matplotlib.pyplot as plt

class Tick_vectors(Tick):
    def __init__(self, data, v_width, v_prominence):
        self.data = data
        self.vol = np.array(self.data['VOL'])
        self.price = np.array(self.data['CLOSE'])
        self.vol = (self.vol - self.vol.min()) / (self.vol.max() - self.vol.min())
        #super().__init__(file)
        print("Creating Tick_vectors")
        self.v_prominence = v_prominence
        self.v_width = v_width
        self.peaks = []
        self.vectors = []
        self.vector_base = []
        self.ema1=self.ema(int(self.size/90))
        self.ema2=self.ema(int(self.size/40))
        self.minmax()

    @classmethod
    def from_file(cls, file, v_width, v_prominence):
        super().__init__(cls, file)
        print(cls.ticker)
        return cls(cls.data, v_width, v_prominence)

    def vector(self, start, width):

        vector = self.vectors[start:start+width,:]
        norm_vector = np.empty_like(vector)
        norm_vector[:, 0] = normalize_list(vector[:, 0]) #x
        norm_vector[:, 1] = normalize_list(vector[:, 1]) #y
        norm_vector[:, 2] = normalize_list(vector[:, 2]) #volume

        return norm_vector

    def plot(self):
        plot_nvector_any(self.vectors)



    def minmax(self):
        from scipy.signal import find_peaks
        from scipy.signal import find_peaks_cwt
        v_prominence = self.v_prominence
        v_width = self.v_width
        df = pd.DataFrame()
        df['CLOSE'] = self.price
        peaks, _ = find_peaks(df['CLOSE'], prominence=v_prominence , width=v_width, distance=v_width*4000)
        #peaks = find_peaks_cwt(df['Close'], np.arange(v_width/2, v_width))
        min = np.zeros(len(peaks)-1, dtype=int)
        ii = range(1, len(peaks))
        vectors = np.zeros([len(peaks)*2 - 1, 3], dtype=float) # -2
        for i in ii:
            range1 = int(peaks[i-1])
            range2 = int(peaks[i])
            min[i-1] = range1 + np.argmin(self.price[range1:range2])
            vectors[i*2 - 2, 0] = self.price[min[i-1]] - self.price[range1]
            vectors[i*2 - 2, 1] = min[i-1] - range1
            vectors[i*2 - 2, 2] = sum(self.vol[range1:min[i-1]])*np.sign(vectors[i*2 - 2, 0])
            vectors[i*2 - 1, 0] = self.price[range2] - self.price[min[i - 1]]
            vectors[i*2 - 1, 1] = range2 - min[i - 1]
            vectors[i*2 - 1, 2] = sum(self.vol[min[i - 1]:range2])*np.sign(vectors[i*2 - 1, 0])

        vectors[-1, 0] = self.price[-1] - self.price[int(peaks[-1])]
        vectors[-1, 1] = self.price.shape[0] - int(peaks[-1])
        vectors[-1, 2] = sum(self.vol[int(peaks[-1]):self.price.shape[0]])*np.sign(vectors[-1, 0])
        vectors[:,2] = vectors[:,2]/np.max(vectors[:,2])*self.price[0]*0.1 #normalizing volume to 10 pct of price

        self.mean_len = np.mean(np.abs(vectors[:,1]))
        self.mean_price_move = np.mean(np.abs(vectors[:,0]))
        peaks = np.concatenate((peaks, min))
        #print('mean length:', self.data['DATE'][int(self.mean_len)] - self.data['DATE'][0])
        print('mean change:', self.mean_price_move)
        print("total vectors:", vectors.shape[0])

        self.peaks = np.sort(peaks)
        #self.peakprice = [self.price[i] for i in self.peaks]
        self.vectors = vectors
        #self.rich_vectors = Vectors(self.vectors, self.price[peaks[0]])
        self.initial_price = self.price[peaks[0]]
        self.last_peak = self.peaks[-1]
        self.first_peak = self.peaks[0]

        return np.sort(peaks), vectors


    def save_vectors(self, file):
        with open(file, 'a') as csvfile:
            np.savetxt(csvfile, self.vectors, delimiter=",")
        print("Vectors saved: ", self.vectors.shape[0])

    def load_vectors(self, file):
        with open(file) as csvfile:
            self.vectors = np.loadtxt(csvfile, delimiter=",")
        print("Vectors loaded: ", self.vectors.shape[0])

    def search_vector(self, vector_target, lag_width, vector_correlation): #plots best vectors
        plot_nvector_any(vector_target, rgb=(0.9, 0, 0))
        for ve in range(0, self.vectors.shape[0] - lag_width):
            checkvec = self.vector(ve, lag_width)
            doublevec = self.vector(ve, lag_width * 3)  # rg = (sber.vectors.shape[0] - ve)/sber.vectors.shape[0]
            if np.sign(vector_target[0, 0]) == np.sign(checkvec[0, 0]):
                i = corr_vectors(vector_target, checkvec, vector_correlation)
            # i = corr_vectors(sber.vector(last_lag, lag_width), sber.vector(ve, lag_width), vector_correlation)
            if i == 1:
                plot_nvector_any(doublevec, divider=lag_width)
                i = 0
        plt.show()

    def search_vectors(self, vector_target, lag_width, vector_correlation): #returns array of vectors
        results = []
        i=0
        #plot_nvector_any(vector_target, rgb=(0.9, 0, 0))
        for ve in range(0, self.vectors.shape[0] - lag_width):
            checkvec = self.vector(ve, lag_width)
            doublevec = self.vector(ve, lag_width * 3)  # rg = (sber.vectors.shape[0] - ve)/sber.vectors.shape[0]
            if np.sign(vector_target[0, 0]) == np.sign(checkvec[0, 0]):
                i = corr_vectors(vector_target, checkvec, vector_correlation)
            # i = corr_vectors(sber.vector(last_lag, lag_width), sber.vector(ve, lag_width), vector_correlation)
            if i == 1:
                results.append(ve)
                if len(results) > 3:
                    return -1
                i=0
        if len(results) == 0:
            return 0
        else:
            return results



def normalize_list(list):
    max = np.abs(list).max()
    list_norm = np.empty_like(list)
    #print("max", max, vector)
    for i in range(0, len(list)):
        list_norm[i] = list[i]/max*10
        #print(vector[i])
    #print(vector)
    return list_norm

def corr_vectors(one, two, vector_correlation): #compares two vectors
    yes = False

    less = vector_correlation
    more = 1/vector_correlation
    for x, y in zip(one, two):
        #print(y[0]*y[2])
        #x[3] len x[4] sin
        leny = math.sqrt(y[0] * y[0] + y[1] * y[1])
        lenx = math.sqrt(x[0] * x[0] + x[1] * x[1])

        siny = abs(y[0]/leny)
        sinx = abs(x[0]/lenx)
        root_vol_y = leny #math.sqrt(abs(leny))
        root_vol_x = lenx #math.sqrt(abs(lenx))
        voly= y[2]
        volx = x[2]
        cond1 = True if np.sign(y[0]) == np.sign(x[0]) else False #sign
        cond2 = True if abs(root_vol_y * less) < abs(root_vol_x) < abs(root_vol_y * more) else False #length
        cond3 = True if siny * less < sinx < siny * more else False #angle
        cond4 = True if voly*less < volx < voly*more else False #volume
        allz = [cond1, cond2, cond3, cond4]

        if all(allz):
            #print(y[0], x[0], leny, lenx)
            #print(less, 1/vector_correlation, y[0], abs(y[0]*float(less)), abs(x[0]), abs(y[0]*float(more)))
            yes = True
        else:
            return 0
    #print('return', y[0], x[0], leny, lenx)
    return 1


def plot_nvector_any(vector, width=1, rgb=(), divider=0):
    x = 0
    y = 0


    if not rgb:
        rgb = (random.random(), random.random(), random.random())

    for i in range(vector.shape[0]):
        x1, y1 = [x, x + vector[i, 0]], [y, y + vector[i, 1]]
        xv, yv = [x, x + np.sign(vector[i, 0]) * vector[i, 2]], [y, y]
        x = x + vector[i, 0]
        y = y + vector[i, 1]
        plt.plot(y1, x1, color=rgb)
        plt.plot(yv, xv, color=rgb)
        if i > 1 and i > divider-1 and divider != 0:
            plt.plot(y1, x1, "xr")

def plot_vector_html(vector, width=1, rgb=(), divider=0, init_price=0):
    import io
    import base64
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    x = init_price
    y = 0
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Prediction")
    axis.grid()


    if not rgb:
        rgb = (random.random(), random.random(), random.random())

    for i in range(vector.shape[0]):
        x1, y1 = [x, x + vector[i, 0]], [y, y + vector[i, 1]]
        xv, yv = [x, x + np.sign(vector[i, 0]) * vector[i, 2]], [y, y]
        x = x + vector[i, 0]
        y = y + vector[i, 1]
        axis.plot(y1, x1, color=rgb)
        axis.plot(yv, xv, color=rgb)
        if i > 1 and i > divider-1 and divider != 0:
            axis.plot(y1, x1, "xr", color=(0.2, 0, 0))


    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    return pngImageB64String







