import random
import numpy as np
from tick import Tick
import pandas as pd
import math
#from vectors import Vectors
import matplotlib.pyplot as plt

class Tick_vectors(Tick):
    def __init__(self, data, v_width=0.005, v_prominence=0.01, add_mean=None, ema=(250,1500), normalize=True):
        self.data = data
        self.normalize_state = normalize
        if add_mean is None:
            self.price = np.array(self.data['CLOSE'])#.rolling(window=200).mean()) #added averaging here
            self.size = data.shape[0]
            self.mean = 30


        else:
            self.price = np.array(self.data['CLOSE'].rolling(window=add_mean).mean()[add_mean:])#added averaging here
            self.vol = np.array(self.data['VOL'][add_mean:])
            self.size = data.shape[0]-add_mean
            self.mean = add_mean #div by 3
            v_width = v_width*add_mean/30
            v_prominence = v_width*2
        print("min", self.mean)

        self.vol = (self.vol - self.vol.min()) / (self.vol.max() - self.vol.min())
        #super().__init__(file)
        print("Creating Tick_vectors")
        if self.price[0] < 0.01:
            self.v_prominence = self.price[0]/50
        elif self.price[0] < 2:
            self.v_prominence = v_prominence / 100
        else:
            self.v_prominence = v_prominence
        self.v_width = v_width
        self.peaks = []
        self.vectors = []
        self.vectors_flat = []
        self.vector_base = []
        self.ema1, self.diff_ema1, self.diff2_ema1 = self.ema(ema[0])
        self.ema2, self.diff_ema2, self.diff2_ema2 = self.ema(ema[1])
        #cutting data to take ema into account
        self.ema1 = self.ema1[ema[1]:]
        self.diff_ema1 = self.diff_ema1[ema[1]:]
        self.diff2_ema1 = self.diff2_ema1[ema[1]:]
        self.ema2 = self.ema2[ema[1]:]
        self.diff_ema2 = self.diff_ema2[ema[1]:]
        self.diff2_ema2 = self.diff2_ema2[ema[1]:]
        self.price = self.price[ema[1]:]
        self.vol = self.vol[ema[1]:]


        self.minmax2()


    @classmethod
    def from_file(cls, file, v_width=0.005, v_prominence=0.01, add_mean=None, ema=(250, 1500), normalize=True):
        super().__init__(cls, file)
        print(cls.ticker, add_mean)

        return cls(cls.data, v_width, v_prominence, add_mean, ema, normalize)

    def vector(self, start, width):

        vector = self.vectors[start:start+width,:]
        norm_vector = np.empty_like(vector)
        norm_vector[:, 0] = normalize_list(vector[:, 0]) if self.normalize_state else vector[:, 0]/100#x
        norm_vector[:, 1] = normalize_list(vector[:, 1]) #y
        norm_vector[:, 2] = normalize_list(vector[:, 2]) #volume
        norm_vector[:, 3] = normalize_list(vector[:, 3]) #ema1
        norm_vector[:, 4] = normalize_list(vector[:, 4]) #ema2
        norm_vector[:, 5] = vector[:, 5] #sin
        norm_vector[:, 6] = vector[:, 6] # diff-ema1
        norm_vector[:, 7] = vector[:, 7] #diff-ema2
        norm_vector[:, 8] = vector[:, 8] #diff2-ema1
        norm_vector[:, 9] = vector[:, 9] #diff2-ema2

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
        min_peaks, _ = find_peaks(-df['CLOSE'], prominence=v_prominence, width=v_width, distance=v_width * 4000)

        if self.comb_ratio is not None:
            peaks = self.comb_peaks(peaks, self.comb_ratio)
        #peaks = find_peaks_cwt(df['Close'], np.arange(v_width/2, v_width))
        print(self.comb_ratio)
        min = np.zeros(len(peaks)-1, dtype=int)
        ii = range(1, len(peaks))
        vectors = np.zeros([len(peaks)*2 - 1, 5], dtype=float) # -2
        for i in ii:
            range1 = int(peaks[i-1])
            range2 = int(peaks[i])
            min[i-1] = range1 + np.argmin(self.price[range1:range2])
            vectors[i*2 - 2, 0] = self.price[min[i-1]] - self.price[range1]
            vectors[i*2 - 2, 1] = min[i-1] - range1
            vectors[i*2 - 2, 2] = sum(self.vol[range1:min[i-1]])*np.sign(vectors[i*2 - 2, 0])
            vectors[i * 2 - 2, 3] = self.price[min[i-1]] - self.ema1[min[i-1]]
            vectors[i * 2 - 2, 4] = self.price[min[i-1]] - self.ema2[min[i-1]]
            vectors[i*2 - 1, 0] = self.price[range2] - self.price[min[i - 1]]
            vectors[i*2 - 1, 1] = range2 - min[i - 1]
            vectors[i*2 - 1, 2] = sum(self.vol[min[i - 1]:range2])*np.sign(vectors[i*2 - 1, 0])
            vectors[i * 2 - 1, 3] = self.price[range2] - self.ema1[range2]
            vectors[i * 2 - 1, 4] = self.price[range2] - self.ema2[range2]

        vectors[-1, 0] = self.price[-1] - self.price[int(peaks[-1])]
        vectors[-1, 1] = self.price.shape[0] - int(peaks[-1])
        vectors[-1, 2] = sum(self.vol[int(peaks[-1]):self.price.shape[0]]) * np.sign(vectors[-1, 0])
        vectors[-1, 3] = self.price[-1] - self.ema1[-1]
        vectors[-1, 4] = self.price[-1] - self.ema2[-1]
        vectors[:, 2] = vectors[:, 2] / np.max(vectors[:, 2]) * self.price[0] * 0.1  # normalizing volume to 10 pct of price
        if vectors[-1, 0]*vectors[-2, 0] > 0 or vectors[-2, 1]/vectors[-1, 1] > 5 :
            vectors[-2, 0]+=vectors[-1, 0]
            vectors[-2, 1]+=vectors[-1, 1]
            vectors = vectors[:-1]


        self.mean_len = np.mean(np.abs(vectors[:,1]))
        self.mean_price_move = np.mean(np.abs(vectors[:,0]))
        self.mean_price = self.price.mean()
        peaks = np.concatenate((peaks, min))
        #print('mean length:', self.data['DATE'][int(self.mean_len)] - self.data['DATE'][0])
        print('mean change:', self.mean_price_move)
        print('mean price:', self.mean_price)
        print("total vectors:", vectors.shape[0])

        self.peaks = np.sort(peaks)
        #self.peakprice = [self.price[i] for i in self.peaks]
        self.vectors = vectors
        #self.rich_vectors = Vectors(self.vectors, self.price[peaks[0]])
        self.initial_price = self.price[peaks[0]]
        self.last_peak = self.peaks[-1]
        self.first_peak = self.peaks[0]

        return np.sort(peaks), vectors

    def minmax2(self):
        from scipy.signal import find_peaks
        from scipy.signal import find_peaks_cwt
        #v_prominence = self.v_prominence
        v_prominence = self.mean/30
        #v_width = self.v_width
        v_width = self.mean
        df = pd.DataFrame()
        df['CLOSE'] = self.price
        peaks, _ = find_peaks(df['CLOSE'], prominence=v_prominence, width=v_width/20, distance=v_width)
        min_peaks, _ = find_peaks(-df['CLOSE'], prominence=v_prominence, width=v_width/20, distance=v_width)

        peaks = np.concatenate((peaks, min_peaks))
        peaks = np.sort(peaks)
        print("total", peaks.shape[0])
        short_one=0
        peaks = self.remove_short(peaks, self.mean*3)
        peaks = self.remove_small(peaks, self.mean)
        peaks = self.remove_short(peaks, self.mean*3)
        peaks = self.remove_small(peaks, self.mean)
        peaks = self.remove_short(peaks, self.mean*3)
        #peaks = self.remove_small(peaks, 5)
        #peaks = self.remove_short(peaks, 20)
        #peaks = self.remove_short(peaks, 100)
        #peaks = self.remove_after_high(peaks, 20)
        #peaks = self.remove_same(peaks)


        print('Maximum vector:', self.find_minmax(peaks))

        #peaks = self.remove_after_high(peaks, 5)
        #peaks = self.remove_after_high(peaks, 8)
        #peaks = self.remove_after_high(peaks, 10)
        #peaks = self.remove_after_high(peaks, 15)
        #peaks = self.remove_after_high(peaks, 20)
        #peaks = self.remove_same(peaks)
        #peaks = self.remove_same(peaks)


        #peaks = self.remove_short(peaks, 80)



        j = 0
        delta = 1
        small_change=0
        #short_one=0
        lil_rel=0
        same=0



        print('Found small change', small_change)
        print('Found short one', short_one)


        j = 0
        delta = 1

        minusone = 0
        while j < len(peaks) - 2:
            vec1 = self.price[peaks[j + 1]] - self.price[peaks[j]]
            vec2 = self.price[peaks[j + 2]] - self.price[peaks[j + 1]]
            #print(vec1, vec2)
            if vec1 * vec2 > 0:
                peaks = np.delete(peaks, j + 1)
                minusone += 1
            else:
                j += 1
        print('-1', minusone)
        print(peaks.shape[0])




        ii = range(0, len(peaks)-2)
        vectors = np.zeros([len(peaks)-2, 10], dtype=float)
        for i in ii:

            vectors[i, 0] = self.price[peaks[i + 1]] - self.price[peaks[i]]
            prct_change = (vectors[i, 0] / self.price[peaks[i]])*100*self.mean*2
            vectors[i, 1] = peaks[i + 1] - peaks[i]
            vectors[i, 2] = sum(self.vol[peaks[i]:peaks[i+1]]) * np.sign(vectors[i, 0])
            vectors[i, 3] = self.price[i] - self.ema1[i]
            vectors[i, 4] = self.price[i] - self.ema2[i]
            vectors[i, 5] = prct_change/np.sqrt((prct_change**2 + vectors[i, 1]**2)) #sin
            if vectors[i, 5] != vectors[i, 5]:
                print('SINUS NONE', i)
            #print('SINIS', vectors[i,5], prct_change, vectors[i, 1])#sin
            vectors[i, 6] = self.diff_ema1[i]
            vectors[i, 7] = self.diff_ema2[i]
            vectors[i, 8] = self.diff2_ema1[i]
            vectors[i, 9] = self.diff2_ema2[i]


        self.mean_len = np.mean(np.abs(vectors[:, 1]))
        self.mean_price_move = np.mean(np.abs(vectors[:, 0]))
        self.mean_price = self.price.mean()
        self.peaks = peaks
        self.vectors = vectors
        # self.rich_vectors = Vectors(self.vectors, self.price[peaks[0]])
        self.initial_price = self.price[peaks[0]]
        self.last_peak = self.peaks[-1]
        self.first_peak = self.peaks[0]
        print('mean change:', self.mean_price_move)
        print('mean price:', self.mean_price)
        print("total vectors:", vectors.shape[0])
        return peaks, vectors
    def remove_same(self, peaks):
        j=0
        lil_rel=0
        same=0

        while j < len(peaks) - 3:
            vec1 = self.price[peaks[j+1]] - self.price[peaks[j]]
            vec2 = self.price[peaks[j+2]] - self.price[peaks[j+1]]
            vec3 = self.price[peaks[j + 3]] - self.price[peaks[j + 2]]

            if abs(vec1)> abs(vec2)*5 or abs(vec2) > abs(vec1)*5:
                peaks = np.delete(peaks, j+2)
                peaks = np.delete(peaks, j+1)
                lil_rel+=2
                j+=2

            elif 0.8<abs(vec1)/abs(vec2)<1.2:
                peaks = np.delete(peaks, j+2)
                peaks = np.delete(peaks, j+1)
                same+=2
                j+=1

            else:
                j += 1
        print('Found lil relation', lil_rel)
        print('Found same', same)
        return peaks

    def remove_after_high(self, peaks, percent):
        j=0
        sum=0

        while j < len(peaks) - 3:
            percent = (self.price[peaks[j]]/100)*(self.mean/10)*3
            vec1 = self.price[peaks[j+1]] - self.price[peaks[j]]
            vec2 = self.price[peaks[j+2]] - self.price[peaks[j+1]]
            vec3 = self.price[peaks[j + 3]] - self.price[peaks[j + 2]]
            if abs(vec1) > percent and abs(vec2) < abs(vec1)/2 and abs(vec3) < abs(vec1)/2:
                peaks = np.delete(peaks, j+3)
                peaks = np.delete(peaks, j+2)
                sum+=1
            else:
                j+=2
        print("Removed after high:", sum)
        return peaks


    def remove_small(self, peaks, percent):
        j=0
        sum=0
        percents = {'1': 1, '5': 1.1, '10': 1.4, '15': 2.5, '30': 6, '60': 10, '120': 15, '240': 20}
        #maxprct, percent = self.find_minmax(peaks)
        #percent = percent * 10

        while j < len(peaks) - 3:
            #percent = (self.price[peaks[j]]/100)*(self.mean/6)
            percent = percents[str(self.mean)]


            vec1 = self.price[peaks[j+1]] - self.price[peaks[j]]
            vec2 = (self.price[peaks[j+2]] - self.price[peaks[j+1]])/self.price[peaks[j+1]]*100
            vec3 = self.price[peaks[j + 3]] - self.price[peaks[j + 2]]
            if abs(vec2) < percent:# and abs(vec2)<abs(vec1) and abs(vec2)<abs(vec3):
                peaks = np.delete(peaks, j+2)
                peaks = np.delete(peaks, j+1)
                #print('ve', vec1, vec2, vec3)
                sum+=1
            else:
                j+=1
        print("Removed small:", sum, percent)
        return peaks

    def remove_short(self, peaks, mult):
        j=1
        sum=0
        while j < len(peaks) - 2:
            vec1 = self.price[peaks[j]] - self.price[peaks[j - 1]]
            vec2 = self.price[peaks[j + 1]] - self.price[peaks[j]]
            vec3 = self.price[peaks[j + 2]] - self.price[peaks[j + 1]]
            if ((peaks[j + 1] - peaks[j]) < self.mean * mult) and (abs(vec2) < abs(vec3)):
                peaks = np.delete(peaks, j + 1)
                peaks = np.delete(peaks, j)
                sum+=1
            else:
                j += 1
        print("Removed short:", sum)
        return peaks
    def find_minmax(self, peaks):
        maxvec = 0
        minvec = abs(self.price[peaks[1]] - self.price[peaks[0]])
        for i, j in zip(peaks, peaks[1:]):
            vec = self.price[j] - self.price[i]
            if abs(vec) > maxvec:
                maxvec=abs(vec)
            if abs(vec) < minvec:
                minvec=abs(vec)

        print("Max, min:", maxvec, minvec)

        return maxvec, minvec


    def comb_peaks(self, peaks, comb_ratio=0.7):
        i = 0
        j = 0
        n = 0

        #delta = (self.size/peaks.shape[0])
        delta=self.mean*20 #to change vector length
        percent=3#self.mean/5
        print(self.size, delta, self.mean)

        while j < len(peaks) - 1:
            min = self.local_min(peaks, j) - peaks[j]
            min1 = peaks[j+1] - self.local_min(peaks, j)
            if peaks[j + 1] - peaks[j] < delta or abs(min-min1)/(min+min1)<0.05:
                peaks = np.delete(peaks, j+1)
            else:
                j += 1

        print('Peaks', peaks.shape)

        j=0

        while j < len(peaks) - 1:
            if abs((self.price[peaks[j + 1]] - self.price[peaks[j]])/self.price[peaks[j]])*100<percent:
                peaks = np.delete(peaks, j + 1)
            else:
                j += 1

        print('Peaks2', peaks.shape)



        while i < len(peaks) - 3:
            if self.comb_check(peaks, i, comb_ratio):
                peaks = np.delete(peaks, i + 1)
            else:
                i += 1
        flats=0

        while n < len(peaks) - 1:
            flatnum = self.is_flat(peaks, n, percent*2)
            if flatnum > 2:
                peaks = np.delete(peaks, [n+p+1 for p in range(1,flatnum-1)])
                flats+=1
                n+=flatnum
            else:
                n+=1
        print("Vectors that are not flat anymore:" , flats)

        return peaks
    def local_min(self, peaks, j):
        range1 = int(peaks[j])
        range2 = int(peaks[j + 1])
        min1 = range1 + np.argmin(self.price[range1:range2])
        return min1

    def comb_check(self, peaks, i, comb_ratio=1):
        range1 = int(peaks[i])
        range2 = int(peaks[i + 1])
        range3 = int(peaks[i + 2])
        range4 = int(peaks[i + 3])


        min1 = range1 + np.argmin(self.price[range1:range2])
        min2 = range2 + np.argmin(self.price[range2:range3])
        min3 = range3 + np.argmin(self.price[range3:range4])
        x1 = self.price[min1] - self.price[range1]
        x2 = self.price[range2] - self.price[min1]
        x3 = self.price[min2] - self.price[range2]
        x4 = self.price[range3] - self.price[min2]
        x5 = self.price[min3] - self.price[range3]
        x6 = self.price[range4] - self.price[min3]

        #if (abs(x2/x1) < comb_ratio and (abs(x3)*1.1 > abs(x2) or abs(x5)>abs(x4))) or (abs(x3/x2) < comb_ratio and (abs(x4)*1.1 > abs(x3) or abs(x6)>abs(x5))  ):
        if (abs(x2/x1) < comb_ratio and (abs(x3)*1.1 > abs(x2))) or (abs(x3/x2) < comb_ratio and (abs(x4)*1.1 > abs(x3))):
            return True
        else:
            return False

    def is_flat(self, peaks, i, tolerance):

        tolerance_percentage = tolerance
        k = 1
        not_flat = True
        while not_flat and i + k < len(peaks):
            corr = (self.price[peaks[i]] - self.price[peaks[i + k]]) / self.price[peaks[i]] * 100
            if np.abs(corr) < tolerance_percentage:
                k = k + 1
            else:
                not_flat = False
        return k


    def flat_check(self):
        i=1
        vectors_flat = self.vectors.copy()
        tolerance_percentage = self.mean_price_move / self.price.min() * 100
        ks = []
        while i < len(self.peaks-3):
            not_flat = True
            k=1
            delta = 0
            delta_len = 0
            delta_vol = 0
            #print(self.peaks[i],  self.price[self.peaks[i]])
            while not_flat and i+k<len(self.peaks):
                corr = (self.price[self.peaks[i]]-self.price[self.peaks[i+k]])/self.price[self.peaks[i]]*100
                if np.abs(corr) < tolerance_percentage:
                    #print(corr, self.price[self.peaks[i]], self.price[self.peaks[i+k]], self.vectors[i+k-1])
                    k = k + 1
                else:
                    not_flat=False

            if k > 3:
                for m in range(k):
                    source_val = self.vectors[i, 0]
                    delta += self.vectors[i + m, 0]
                    #delta_len += self.vectors[i + m, 1]
                    vectors_flat[i + m, 0] = 0
                    if m > 0 and m < k-1:
                        ks.append(i + m)
                        delta_len += self.vectors[i + m, 1]
                        delta_vol += abs(self.vectors[i + m, 2])

            i += k
            vectors_flat[i-1, 0] += delta
            vectors_flat[i-k, 1] += delta_len
            vectors_flat[i-k, 2] += delta_vol

        vectors_flat = np.delete(vectors_flat, ks, 0)
        self.peaks = np.delete(self.peaks, ks, 0)
        print("Total flattened vectors:", vectors_flat.shape[0])
        plot_nvector_any(vectors_flat)
        #return vectors_flat








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
    list_norm = (list/max)
    #print("max", max, vector)
    #for i in range(0, len(list)):
    #    list_norm[i] = list[i]/max*10
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
        #plt.plot(yv, xv, color=rgb)
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







