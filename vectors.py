import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
from tick_vectors import Tick_vectors


class Vectors():
    def __init__(self, vectors, init_price, peaks=None):
        print("Initial price 0: ", init_price)
        self.vectors = vectors
        self.length = len(self.vectors)
        self.first_peak = 0 if peaks is None else int(peaks[0])
        self.last_peak = 0 if peaks is None else int(peaks[-1])
        self.peaks = peaks
        self.x_max = 0
        self.y_max = 0
        self.v_max = 0
        self.init_price = init_price
        self.normalized = self.normalize()
        print("Total vectors: ", len(vectors))
        print("Initial price: ", init_price)

    @classmethod
    def from_file(cls, file, v_width, v_prominence):
        tvectors = Tick_vectors.from_file(file, v_width, v_prominence)
        first_price = tvectors.price[int(tvectors.peaks[0])]
        #cls.last_price = tvectors.price[int(tvectors.peaks[-1])]
        peaks = tvectors.peaks

        return cls(tvectors.vectors, first_price, peaks)

    @classmethod
    def from_df(cls, data, v_width, v_prominence):
        tvectors = Tick_vectors(data, v_width, v_prominence)
        first_price = tvectors.price[int(tvectors.peaks[0])]
        #cls.last_price = tvectors.price[int(tvectors.peaks[-1])]
        peaks = tvectors.peaks

        return cls(tvectors.vectors, first_price, peaks)

    def add(self, x, y, v):
        self.vectors = np.concatenate([self.vectors, np.array([[float(x), float(y), float(v)]], dtype=object)])
        self.normalized = self.normalize()
        #self.vectors = np.concatenate([self.vectors, np.array([[float(x*self.x_max/10), float(y*self.y_max/10), float(v*self.v_max/10)]], dtype=object)])



    def slice(self, start, width):
        price = self.init_price
        for i in range(start):
            price += self.vectors[i,0]

        vector = self.vectors[start:start+width,:]
        if self.peaks is not None:
            peaks = self.peaks[start:start+width]
        else:
            peaks = None

        #norm_vector = np.empty_like(vector)
        #print('vector', vector[:,0], vector[:,1], vector[:,2])
        #norm_vector[:, 0] = normalize_list(vector[:, 0]) #x
        #norm_vector[:, 1] = normalize_list(vector[:, 1]) #y
        #norm_vector[:, 2] = normalize_list(vector[:, 2]) #volume

        #print('normed_vector', norm_vector[:, 0], norm_vector[:, 1], norm_vector[:, 2])
        #return norm_vector[:,:3]
        return Vectors(vector, price, peaks)

    def normalize(self):
        norm_vector = np.empty_like(self.vectors)
        # print('vector', vector[:,0], vector[:,1], vector[:,2])
        norm_vector[:, 0], self.x_max = normalize_list(self.vectors[:, 0]) #x
        norm_vector[:, 1], self.y_max = normalize_list(self.vectors[:, 1]) #y
        norm_vector[:, 2], self.v_max = normalize_list(self.vectors[:, 2])#volume

        return norm_vector

    def comb(self, ratio=0.9):
        neu = np.zeros((0,3))
        print(neu)
        n=True
        i=0
        while i<(len(self.vectors)-3):
            #print(self.vectors[i + 1, 0], self.vectors[i, 0], np.abs(self.vectors[i + 1, 0] / self.vectors[i, 0]))
            j=0
            if np.abs(self.vectors[i+1,0]/self.vectors[i,0])<ratio:
                combinedx = self.vectors[i, 0]
                combinedy = self.vectors[i, 1]
                combinedv = self.vectors[i, 2]
                keep_running = True
                while np.abs(self.vectors[i+j+1,0]) < np.abs(self.vectors[i+j+2,0])*ratio and keep_running:
                    combinedx+= (self.vectors[i+j+1,0]+self.vectors[i+j+2,0])
                    combinedy+=(self.vectors[i+j+1,1]+self.vectors[i+j+2,1])
                    combinedv+=(np.abs(self.vectors[i+1+j,2])+np.abs(self.vectors[i+2+j,2]))*np.sign(self.vectors[i, 2])
                    j += 2
                    if i+j>len(self.vectors)-5:
                        break

                neu = np.concatenate((neu, np.array([[combinedx,combinedy,combinedv]])), axis=0)
                i+=3+j-2
            else:
                neu = np.concatenate((neu, np.array([[self.vectors[i,0],self.vectors[i,1],self.vectors[i,2]]])), axis=0)
                i+=1

        last_piece = len(self.vectors) - i
        print('Last piece', last_piece)
        for i in range(last_piece,0,-1):
            neu = np.concatenate((neu, np.array([[self.vectors[-i, 0], self.vectors[-i, 1], self.vectors[-i, 2]]])), axis=0)
        peaks = []
        sum=self.peaks[0]
        for peak in neu[:,1]:
            peaks.append(int(sum+peak))
            sum+=peak
        print(self.peaks, len(self.peaks))
        print(peaks,  len(peaks))


        #self.plot()
        return Vectors(neu, self.init_price, peaks)
        #print(neu)







    def px(self, first_peak=0):
        self.data = pd.DataFrame()
        close = []
        ind=[]
        x=self.init_price
        y=first_peak
        for i in range(self.vectors.shape[0]):
            x+=self.vectors[i,0]
            y+=self.vectors[i,1]
            close.append(x)
            ind.append(y)
        self.data['Y'] = close
        self.data["X"] = ind
        return self.data






    def plot(self, width=1, rgb=(), divider=0):
        x = self.init_price
        y = 0

        #fig, ax = plt.subplots(1)
        #ax.grid(True)

        if not rgb:
            rgb = (random.random(), random.random(), random.random())

        for i in range(self.vectors.shape[0]):

            x1, y1 = [x, x + self.vectors[i, 0]], [y, y + self.vectors[i, 1]]
            xv, yv = [x, x + np.sign(self.vectors[i, 0]) * self.vectors[i, 2]/5], [y, y]
            x = x + self.vectors[i, 0]
            y = y + self.vectors[i, 1]
            plt.plot(y1, x1, color=rgb)
            #ax.plot(y1, x1, color=rgb)
            plt.plot(yv, xv, color=rgb)
            #ax.plot(yv, xv, color=rgb)
            if i > 1 and i > divider - 1 and divider != 0:
                plt.plot(y1, x1, "xr")
                #ax.plot(y1, x1, "xr")

    def plot_html(self, width=1, rgb=(), divider=0):
        import io
        import base64
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        x = self.init_price
        y = 0
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title("Prediction")
        axis.grid()

        if not rgb:
            rgb = (random.random(), random.random(), random.random())

        for i in range(self.vectors.shape[0]):
            x1, y1 = [x, x + self.vectors[i, 0]], [y, y + self.vectors[i, 1]]
            xv, yv = [x, x + np.sign(self.vectors[i, 0]) * self.vectors[i, 2]/5], [y, y]
            x = x + self.vectors[i, 0]
            y = y + self.vectors[i, 1]
            axis.plot(y1, x1, color=rgb)
            axis.plot(yv, xv, color=rgb)
            if i > 1 and i > divider - 1 and divider != 0:
                axis.plot(y1, x1, "xr", color=(0.2, 0, 0))

        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        return pngImageB64String

    def fast_predict(self, model_file_name, lag_length=10, mode='std'):
        import joblib
        sci_model = joblib.load(model_file_name)
        if mode == 'cos':
            X_p, x_m = self.vector_to_predict_cos(lag_length)

        elif mode == 'sum':
            X_p, x_m = self.vector_to_predict_sum(lag_length)

        elif mode == 'sum2':
            X_p, x_m = self.vector_to_predict_sum2(lag_length)

        elif mode == 'sum2v':
            X_p, x_m = self.vector_to_predict_sum2v(lag_length)

        elif mode == 'std':
            X_p, x_m = self.vector_to_predict_std(lag_length)
        pred = sci_model.predict(X_p)*x_m/10
        return pred

    def get_prediction(self, v_length=20, mode='sum2v', type='gb'):
        total_length = v_length + 3
        print(f'trs_model-1step_{total_length}_{mode}.pkl')
        pred1 = self.fast_predict(f'{type}_model-1step_{total_length}_{mode}.pkl', v_length, mode=mode)
        pred2 = self.fast_predict(f'{type}_model-2step_{total_length}_{mode}.pkl', v_length, mode=mode)
        pred3 = self.fast_predict(f'{type}_model-3step_{total_length}_{mode}.pkl', v_length, mode=mode)

        mean_y = self.vectors[-v_length:, 1].mean()
        mean_v = self.vectors[-v_length:, 2].mean()
        self.add(pred1, mean_y, mean_v)
        self.add(pred2, mean_y, mean_v)
        self.add(pred3, mean_y, mean_v)
        return vector


    def vector_to_predict_std(self, lag_length=10):
        #to_model = pd.DataFrame()
        to_model = prep(['X', 'LEN', 'POWER', 'RATIO', 'Y'], lag_length)
        last_lag = self.vectors.shape[0] - lag_length
        #vector = self.vector(last_lag, lag_length)
        r_vector = self.slice(last_lag-1, lag_length+1)
        x_m = r_vector.x_max
        y_m = r_vector.y_max
        v_m = r_vector.v_max
        vector = r_vector.normalized
        #print("V to pred ", vector)
        temp = pd.DataFrame()

        li = np.zeros(0)

        for j in range(lag_length):

            x = vector[j, 0]
            #x_prev = vector[j - 1, 0]

            y = vector[j, 1]
            power = vector[j, 2] / vector[j, 0]
            ratio = x*abs(power) if vector[j, 2] > 7 else 0

            len = np.sqrt(x**2 + y**2)
            #ratio = x/abs(x_prev)

            li = np.append(li, [x, len, power, ratio, y])

        X_p = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
           # to_model = pd.concat([to_model, temp], ignore_index=True)
           # X_p = pd.DataFrame(vector.reshape(1, -1), columns=to_model.columns)

        return X_p, x_m

    def vector_to_predict_sum(self, lag_length=10):
        #to_model = pd.DataFrame()
        to_model = prep(['X', 'SUM', 'Y'], lag_length)
        last_lag = self.vectors.shape[0] - lag_length*2
        # vector = self.vector(last_lag, lag_length)
        r_vector = self.slice(last_lag, lag_length*2)

        li = np.zeros(0)
        for j in range(lag_length, lag_length*2):
            sum = 0
            sum2 = 0
            x = r_vector.normalized[j, 0]
            y = r_vector.normalized[j, 1]
            for k in range(0, lag_length):
                sum += r_vector.normalized[j - lag_length + k, 0]
            li = np.append(li, [x, sum, y])

        x_m = r_vector.x_max
        y_m = r_vector.y_max
        v_m = r_vector.v_max
        vector = r_vector.normalized
        # print("V to pred ", vector)

        X_p = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
        print(X_p)
        return X_p, x_m


    def vector_to_predict_sum2(self, lag_length=10):
        #to_model = pd.DataFrame()
        to_model = prep(['X', 'SUM1', 'SUM2', 'Y'], lag_length)
        last_lag = self.vectors.shape[0] - lag_length*3
        # vector = self.vector(last_lag, lag_length)
        r_vector = self.slice(last_lag, lag_length*3)

        li = np.zeros(0)
        for j in range(lag_length*2, lag_length*3):
            sum = 0
            sum2 = 0
            x = r_vector.normalized[j, 0]
            y = r_vector.normalized[j, 1]
            for k in range(0, lag_length):
                sum += r_vector.normalized[j - lag_length + k, 0]

            for k2 in range(0, lag_length*2):
                sum2 += r_vector.normalized[j - lag_length*2 + k2, 0]


            li = np.append(li, [x, sum, sum2, y])

        x_m = r_vector.x_max
        y_m = r_vector.y_max
        v_m = r_vector.v_max
        vector = r_vector.normalized
        # print("V to pred ", vector)

        X_p = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
        print(X_p)
        return X_p, x_m

    def vector_to_predict_sum2v(self, lag_length=10):
        #to_model = pd.DataFrame()
        to_model = prep(['X', 'SUM1', 'SUM2', 'POWER', 'LEN', 'RATIO', 'Y'], lag_length)
        last_lag = self.vectors.shape[0] - lag_length*3
        # vector = self.vector(last_lag, lag_length)
        r_vector = self.slice(last_lag, lag_length*3)

        li = np.zeros(0)
        for j in range(lag_length*2, lag_length*3):
            sum = 0
            sum2 = 0
            x = r_vector.normalized[j, 0]
            x_prev = r_vector.normalized[j-1, 0]

            y_prev = r_vector.normalized[j-1, 1]
            power = r_vector.normalized[j, 2]
            y = x * power if power > 7 else x
            vlen = math.sqrt(x ** 2 + y ** 2)
            vratio = x / x_prev
            for k in range(0, lag_length):
                sum += r_vector.normalized[j - lag_length + k, 0]

            for k2 in range(0, lag_length*2):
                sum2 += r_vector.normalized[j - lag_length*2 + k2, 0]


            li = np.append(li, [x, sum, sum2, power, vlen, vratio, y])
            #li = np.append(li, [x, sum, sum2, power, y])

        x_m = r_vector.x_max
        y_m = r_vector.y_max
        v_m = r_vector.v_max
        vector = r_vector.normalized
        # print("V to pred ", vector)

        X_p = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
        print(X_p)
        return X_p, x_m

def vector_to_predict_cos(self, lag_length=10):
        #to_model = pd.DataFrame()
        to_model = prep(['X', 'L', 'COS'], lag_length)
        last_lag = self.vectors.shape[0] - lag_length
        #vector = self.vector(last_lag, lag_length)
        r_vector = self.slice(last_lag, lag_length)

        li = np.zeros(0)
        for j in range(lag_length):

            x = self.normalized[j,0]
            l = math.sqrt(self.normalized[j,0]**2+self.normalized[j,1]**2)
            cos = x/l
            li = np.append(li, [x, l, cos])


        x_m = r_vector.x_max
        y_m = r_vector.y_max
        v_m = r_vector.v_max
        vector = r_vector.normalized
        #print("V to pred ", vector)

        X_p = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
        print(X_p)
        return X_p, x_m





def normalize_list(list):
    max = np.abs(list).max()
    list_norm = np.empty_like(list)
    #print("max", max, vector)
    for i in range(0, len(list)):
        list_norm[i] = list[i]/max*10
        #print(vector[i])
    #print(vector)
    return list_norm, max

def denormalize_list(list_norm, max):
    list = np.empty_like(list_norm)
    for i in range(0, len(list_norm)):
        list[i] = list_norm[i]*max/10

    return list

def denormalize_vector(vector, x, y, v):
    denorm_vector = np.empty_like(vector)
    denorm_vector[:, 0] = denormalize_list(vector[:, 0], x)  # x
    denorm_vector[:, 1] = denormalize_list(vector[:, 1], y)  # y
    denorm_vector[:, 2] = denormalize_list(vector[:, 2], v)  # volume

    return denorm_vector

def prep(features, lag_length=10):
    names = []
    for j in range(lag_length):
        for f in features:
            names.append(f'{f}_{j}')
            # to_model[f'{f}_{j}'] = []
    to_model = pd.DataFrame(columns=names)
    return to_model