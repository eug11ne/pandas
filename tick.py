import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import date
from datetime import datetime
from dateutil.rrule import rrule, DAILY


class Tick():
    def __init__(self, file, v_width=10):
        names = ['TICKER','PER','DATE','TIME','CLOSE','VOL']
        #self.data = np.genfromtxt(file, delimiter=';')
        self.data = pd.read_csv(file, delimiter=',', names=names)
        print("Columns: " + self.data.columns)
        self.data['DATETIME'] = self.data['DATE'].astype(str) + self.data['TIME'].astype(str)
        self.data['DATETIME'] = pd.to_datetime(self.data['DATETIME'], format='%Y%m%d%H:%M')
        print('Extracted: ' + self.data.iloc[0]['TICKER'])
        self.start = self.data['DATETIME'].iloc[0]
        self.end = self.data['DATETIME'].iloc[-1]
        self.size = self.data.shape[0]
        self.target_correlation = 0.85


        self.data.set_index('DATETIME', inplace=True, drop=False)
        print(self.data.index[1], self.data.index[-1])
        self.meanvol = self.data.resample("M").mean()['VOL']
        self.dayvol = self.data.groupby(self.data.index.minute).mean()['VOL']
        self.meanclose = self.data.groupby(self.data.index.hour).mean()['CLOSE']
        self.minclose = self.data.groupby(self.data.index.hour).min()['CLOSE']
        self.maxvol = self.data.groupby(self.data.index.hour).max()['VOL']
        #print("Mean volume: %s day volume: %s, max volume: %s" % (self.meanvol, self.dayvol, self.maxvol))
        print(self.data.index.freq)
        #self.v_prominence = v_prominence
        self.v_width = v_width
        self.peaks = []
        self.vectors = []
        self.vector_base = []

        self.model = pd.DataFrame()


        self.vol = np.array(self.data['VOL'])
        self.price = np.array(self.data['CLOSE'])
        self.vol = (self.vol - self.vol.min()) / (self.vol.max() - self.vol.min())
        self.mean_len = 0
        self.mean_price_move = 0




        if 0 <= self.price.max() <= 10:
            print('more thtan')
            self.price = self.price*100
        else:
            if 0 <= self.price.max() <= 60:
                self.price = self.price * 10
        self.minmax()


    def sumvol(self):
        price = pd.DataFrame()
        df = pd.DataFrame()
        price['Price'] = np.unique(self.price.astype(int))
        price['Vol'] = 0
        price.set_index('Price', inplace=True)
        df['Close'] = self.price
        df['Vol'] = self.vol
        for i, row in df.iterrows():
            price.loc[int(row[0]), 'Vol'] += row[1]
        price.sort_index(inplace=True)
        return price

    def vector(self, start, width):

        vector = self.vectors[start:start+width,:]
        norm_vector = np.empty_like(vector)
        #print('vector', vector[:,0], vector[:,1], vector[:,2])
        norm_vector[:, 0] = normalize_list(vector[:, 0]) #x
        norm_vector[:, 1] = normalize_list(vector[:, 1]) #y
        norm_vector[:, 2] = normalize_list(vector[:, 2]) #volume

        #print('normed_vector', norm_vector[:, 0], norm_vector[:, 1], norm_vector[:, 2])
        return norm_vector[:,:3]

    def save_vectors(self):
        with open('vectors.csv', 'a') as csvfile:
            np.savetxt(csvfile, self.vectors, delimiter=",")


    def load_vectors(self):
        with open('vectors.csv') as csvfile:
            self.vectors = np.loadtxt(csvfile, delimiter=",")
            #base = np.loadtxt(csvfile, delimiter=",")
            #self.vectors = base
            #len = [np.sqrt(x[0]*x[0] + x[1]*x[1]) for x in base]
            #sin = base[:, 0]/len
            #print('sin', sin)

            #self.vector_base = np.c_[base, len, sin]
            print('loaded vectors: ', self.vectors.shape[0])
            #print('vector_base:', self.vector_base)
        #with open('vectors_base.csv', 'a') as csvfilesave:
            #np.savetxt(csvfilesave, self.vector_base, delimiter=",")


    #def minmax(self, v_prominence=1, v_width=10):
    def minmax(self, v_width=10):
        #from scipy.signal import find_peaks
        from scipy.signal import find_peaks_cwt
        #v_prominence = self.v_prominence
        v_width = self.v_width
        df = pd.DataFrame()
        df['Close'] = self.price
        #peaks, _ = find_peaks(df['Close'], prominence=v_prominence, width=v_width)
        peaks = find_peaks_cwt(df['Close'], np.arange(v_width/2, v_width))
        min = np.zeros(len(peaks)-1, dtype=int)
        #print("peak")
        #print(peaks)
        ii = range(1, len(peaks))
        vectors = np.zeros([len(peaks)*2 - 2, 3], dtype=float) # -2
        for i in ii:
            range1 = int(peaks[i-1])
            range2 = int(peaks[i])
            min[i-1] = range1 + np.argmin(self.price[range1:range2])
            vectors[i*2 - 2, 0] = self.price[min[i-1]] - self.price[range1]
            vectors[i*2 - 2, 1] = min[i-1] - range1
            vectors[i*2 - 2, 2] = sum(self.vol[range1:min[i-1]])
            vectors[i*2 - 1, 0] = self.price[range2] - self.price[min[i - 1]]
            vectors[i*2 - 1, 1] = range2 - min[i - 1]
            vectors[i*2 - 1, 2] = sum(self.vol[min[i - 1]:range2])

        self.mean_len = np.mean(np.abs(vectors[:,1]))
        self.mean_price_move = np.mean(np.abs(vectors[:,0]))
        peaks = np.concatenate((peaks, min))
        print("mean")
        print('mean length ')
        print(self.data['DATE'][int(self.mean_len)] - self.data['DATE'][0])
        print('mean change ')
        print(self.mean_price_move)
        print("total vectors: ", vectors.shape[0])
        print(self.data['DATE'][60],  self.data['DATE'][1])
        self.peaks = np.sort(peaks)
        self.vectors = vectors

        return np.sort(peaks), vectors

    def set_interval(self, start, finish):
        #startdate = datetime.strptime(start, '%Y%m%d')
        #enddate = datetime.strptime(finish, '%Y%m%d')
        print(start, finish)
        so=str(start)
        print(so)

        self.slice = self.data.truncate(before=start, after=finish)
        self.start = pd.to_datetime(start, format='%Y%m%d')
        self.end = pd.to_datetime(finish, format='%Y%m%d')
        self.vol = np.array(self.slice['VOL'])
        self.price = np.array(self.slice['CLOSE'])
        self.vol = (self.vol - self.vol.min()) / (self.vol.max() - self.vol.min())
        self.mean_len = 0
        self.mean_price_move = 0
        self.size = self.slice.shape[0]

        if 0 <= self.price.max() <= 10:
            print('more thtan')
            self.price = self.price * 100
        else:
            if 0 <= self.price.max() <= 60:
                self.price = self.price * 10

    def save4prophet(self):
        saving = pd.DataFrame()
        #saving['DT'] = self.data['DATETIME']
        saving['y'] = self.data['CLOSE']
        #saving['VL'] = self.data['VOL']
        print("DF to be saved")
        print(self.data['DATETIME'].astype(str))
        saving.to_csv('cities.csv')

    def turning_points(self, peaks):
        min = pd.DataFrame()
        mm = pd.DataFrame()

        min['MIN'] = np.around(self.price[peaks]%1, 2)
        min['PEAK'] = self.price[peaks]
        min['HIGHER'] = False
        print(min)

        for num, val in enumerate(min['PEAK'][:-1]):
            print(val, min['PEAK'][num+1])
            if val > min['PEAK'][num+1]:
                min.at[num, 'HIGHER'] = True



        mm['FRA'] = np.unique(min['MIN'])
        mm['FREMIN'] = 0
        mm['FREMAX'] = 0
        mm.set_index('FRA', inplace=True)
        mm.sort_index(inplace=True)

        min['FREMIN'] = np.zeros(len(min))
        min['FREMAX'] = np.zeros(len(min))
        # print(min['FRE'])
        # min.set_index('MIN', inplace=True)
        for num, i in enumerate(min['MIN']):
            print(i)
            if min['HIGHER'][num]:
                mm.at[i, 'FREMAX'] += 1
            else:
                mm.at[i, 'FREMIN'] += 1
        #print(mm)
        print(min['HIGHER'])
        return mm

    def modelling(self, pred_len):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import scale
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import mean_absolute_error, r2_score, max_error
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix

        print(f"Modelling  {self.data.iloc[1]['TICKER']}...")
        datasci = pd.DataFrame()
        datasci['CLOSE'] = self.data['CLOSE']
        datasci['VOL'] = self.data['VOL']
        datasci['MEAN'] = datasci['CLOSE'].rolling(window=100).mean()

        for i in range(0, 10):
           datasci[f"CLOSE_{i}"] = datasci['CLOSE'].shift(i)
        learn_len = pred_len
        datasci["TARGET"] = datasci["CLOSE"].shift(-learn_len)
        datasci.reset_index(inplace=True)
        datasci.drop("DATETIME", axis=1, inplace=True)
        datasci.fillna(method="backfill", inplace=True)


        X = datasci[:-learn_len].drop("TARGET", axis=1)
        X_p = datasci[-learn_len:].drop("TARGET", axis=1)

        y = datasci[:-learn_len]["TARGET"]


        print(f"Making prediction for {self.data.iloc[1]['TICKER']}...")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sci_model = RandomForestRegressor(n_estimators=1000, max_depth=10, criterion='squared_error')
        #sci_model = RandomForestRegressor(n_estimators=n_estimators1, max_depth=max_depth1, criterion=criterion1)
        sci_model.fit(X_train, y_train)
        y_pred = sci_model.predict(X_p)
        #y_pred = sci_model.predict(X)
        print("******model")
        print(X_p)
        print(sci_model.score(X_test, y_test))
        print(len(y_pred))
        #plt.plot(y_pred)
        print("end model")
        #e = np.array(datasci['CLOSE'])
        e = np.zeros(len(datasci['CLOSE'])+len(y_pred))
        e[:(len(datasci))] = np.nan
        e[(len(datasci)):] = y_pred
        #cross_validation(sci_model, X[:100], y[:100])

        plt.plot(e)
        plt.plot(datasci[-(learn_len*7):]['CLOSE'])
        #plt.plot(y_pred) #X_train.sort_index())
        #sci_model = RandomForestRegressor(n_estimators=50, max_depth=10, criterion='squared_error')
        #sci_model.fit(X[:500], y[:500])
        plt.show()
        return e

    def create_training_set(self, lag_length, steps=1):
        for j in range(lag_length):
            self.model[f'X_{j}'] = []
            self.model[f'Y_{j}'] = []
            self.model[f'VOL_{j}'] = []
        print(self.model)
        for i in range(10000):
            temp = pd.DataFrame(self.vector(i, lag_length).reshape(1, -1), columns=self.model.columns)
            self.model = pd.concat([self.model, temp],ignore_index=True)

        print("training_set", self.model)
        print("yay")
        column_numbers = [x for x in range(self.model.shape[1])]
        print(column_numbers)
        del column_numbers[-9:]
        print(column_numbers)
        col1 = f"X_{lag_length-1}"
        col2 = f"Y_{lag_length-1}"
        col3 = f"VOL_{lag_length-1}"
        #X = self.model.drop(columns=[col1, col2, col3])
        X = self.model.iloc[:,column_numbers]
        print(X)
        #y = pd.DataFrame()
        y1 = self.model.iloc[:,-9]
        y2 = self.model.iloc[:,-6]
        y3 = self.model.iloc[:,-3]
        y_y1 = self.model.iloc[:,-8]
        y_v1 = self.model.iloc[:,-7]
        print(y1, y2, y3)
        return X, y1, y2, y3, y_y1, y_v1




        #for i in range(len(self.vectors)):
        #    pd.DataFrame(self.vector(i, lag_length).reshape(1,lag_length*3) #, columns=['X','Y','VOL'])]) pd.concat([self.model,
        #print(self.model)




    def predict_vectors(self, vector, pred_len):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import scale
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import mean_absolute_error, r2_score, max_error
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix

        print(f"Modelling vectors for {self.data.iloc[1]['TICKER']}...")
        datasci = pd.DataFrame()
        v_length = len(vector)
        X, y1, y2, y3, y_y1, y_v1 = self.create_training_set(v_length+3)
        X_p = pd.DataFrame(vector.reshape(1, -1), columns=X.columns)

        ''' datasci['X'] = self.vectors[-10000:,0]
        datasci['Y'] = self.vectors[-10000:,1]
        datasci['L'] = (datasci['X']*datasci['X'] + datasci['Y']*datasci['Y'])**(1/2)
        datasci['COS'] = datasci['X']/datasci['L']
        datasci['VOL'] = self.vectors[-10000:,2]
        for ii in range(0,20):
            datasci[f'X_{ii}'] = datasci['X'].shift(ii)
        X_p['X'] = vector[:, 0]
        X_p['Y'] = vector[:, 1]
        X_p['L'] = (X_p['X'] * X_p['X'] + X_p['Y'] * X_p['Y'])**(1/2)
        X_p['COS'] = X_p['X'] / X_p['L']
        X_p['VOL'] = vector[:, 2]
        for jj in range(0,20):
            X_p[f'X_{jj}'] = X_p['X'].shift(jj)
        X_p = X_p.iloc[-2:,:]



        #for i in range(0, 5):
        #    datasci[f"X_{i}"] = datasci['X'].shift(i)
        #    X_p[f"X_{i}"] = X_p['X'].shift(i)
        learn_len = pred_len
        datasci["TARGET"] = datasci["X"].shift(-learn_len)
        datasci.reset_index(inplace=True, drop=True)
        #datasci.drop("DATETIME", axis=1, inplace=True)
        datasci.fillna(value=0, inplace=True)
        X_p.fillna(value=0, inplace=True)

        X = datasci[:-learn_len].drop("TARGET", axis=1)
        print(X)
        print(X_p)
        #X_p = datasci[-learn_len:].drop("TARGET", axis=1)

        y = datasci[:-learn_len]["TARGET"]

        print(f"Making prediction for {self.data.iloc[1]['TICKER']}...")

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'''
        train_len = 9000
        test_len = -1000
        X_train = X[:train_len]
        y1_train= y1[:train_len]
        y2_train = y2[:train_len]
        y3_train = y3[:train_len]
        y_y1_train = y_y1[:train_len]
        y_v1_train = y_v1[:train_len]
        X_test = X[test_len:]
        y1_test = y1[test_len:]
        y2_test = y2[test_len:]
        y3_test = y3[test_len:]
        y_y1_test = y_y1[test_len:]
        y_v1_test = y_v1[test_len:]
        sci_model = RandomForestRegressor(n_estimators=300, max_depth=10, criterion='squared_error')
        # sci_model = RandomForestRegressor(n_estimators=n_estimators1, max_depth=max_depth1, criterion=criterion1)
        #cross_validation(sci_model, X_train, y1_train)
        sci_model.fit(X_train, y1_train)
        y_pred1 = sci_model.predict(X_p)
        # y_pred = sci_model.predict(X)
        print("******model")
        print(X_p)
        print("Model score:", sci_model.score(X_test, y1_test))
        print("Prediction for 1 steps ahead", pred_len, y_pred1)
        sci_model.fit(X_train, y2_train)
        y_pred2 = sci_model.predict(X_p)
        print("Model score:", sci_model.score(X_test, y2_test))
        print("Prediction for 2 steps ahead", pred_len, y_pred2)
        sci_model.fit(X_train, y3_train)
        y_pred3 = sci_model.predict(X_p)
        print("Model score:", sci_model.score(X_test, y3_test))
        print("Prediction for 3 steps ahead", pred_len, y_pred3)

        sci_model.fit(X_train, y_y1_train)
        y_y1_pred = sci_model.predict(X_p)
        # y_pred = sci_model.predict(X)
        print("Model score:", sci_model.score(X_test, y_y1_test))
        print("Prediction for y1", pred_len, y_y1_pred)

        sci_model.fit(X_train, y_v1_train)
        y_v1_pred = sci_model.predict(X_p)
        # y_pred = sci_model.predict(X)
        print("Model score:", sci_model.score(X_test, y_v1_test))
        print("Prediction for v1", pred_len, y_v1_pred)

        # plt.plot(y_pred)
        print("end model")
        print("old", vector)
        vector = np.concatenate([vector, np.array([[float(y_pred1), float(y_y1_pred), float(y_v1_pred)]], dtype=object)])
        vector = np.concatenate([vector, np.array([[float(y_pred2), 2, 2]], dtype=object)])
        vector = np.concatenate([vector, np.array([[float(y_pred3), 2, 2]], dtype=object)])
        #vector = np.concatenate([vector, np.array([y_pred2, 1, 1]).reshape(1, 3)])


        print("prediction", vector)
        plot_nvector_any(vector, divider=v_length)



        # e = np.array(datasci['CLOSE'])
        #e = np.zeros(len(datasci['CLOSE']) + len(y_pred))
        #e[:(len(datasci))] = np.nan
        #e[(len(datasci)):] = y_pred
        # cross_validation(sci_model, X[:100], y[:100])

        #plt.plot(e)
        #plt.plot(datasci[-(learn_len * 7):]['CLOSE'])
        # plt.plot(y_pred) #X_train.sort_index())
        # sci_model = RandomForestRegressor(n_estimators=50, max_depth=10, criterion='squared_error')
        # sci_model.fit(X[:500], y[:500])
        #plt.show()
        return 0

        #plt.show()

    def promodelling(self):
        from prophet import Prophet
        import matplotlib.pyplot as plt
        import pandas as pd

        datasci = pd.DataFrame()
        datasci.reset_index(inplace=True)
        datasci['ds'] = self.data['DATETIME']
        datasci['y'] = self.data['CLOSE']
        #datasci.drop('DATETIME', axis=1, inplace=True)
        m = Prophet().fit(datasci)
        future = m.make_future_dataframe(periods=100, freq='B')
        forecast = m.predict(future)
        fig1 = m.plot(forecast)
        fig2 = m.plot_components(forecast)
        plt.show()

    def enrich(self, tick, col_name):
        print(f"Adding {col_name} to {self.data.iloc[1]['TICKER']}")

        for dtime in self.data['DATETIME']:

            if tick.data.index.isin([dtime]).any():
               self.data.at[dtime,col_name] = tick.data.loc[dtime]['CLOSE']

        print('aaa')

    def corr_price(self, lag_start, lag_length, target_correlation):


        #target_correlation = 0.93
        got_it = False

        max_length = self.slice.shape[0]

        alls = self.slice['CLOSE'].reset_index(drop=True)
        last_lag = alls.truncate(lag_start, lag_start+lag_length,  copy=True).reset_index(
            drop=True)
        #print(last_lag)
        cor = np.zeros(max_length)
        results = np.zeros((0, 3))

        for i in range(0, max_length - lag_length, 70):
            if lag_start-20 < i < lag_start+lag_length+20:
                continue
            cor[i] = last_lag.corr(
                alls.truncate(i, i + lag_length, copy=True).reset_index(drop=True))
            if cor[i] > self.target_correlation:
                print("cori", cor[i])
            results = np.append(results, [[cor[i], i, lag_length]], axis=0)
        to_ret = get_max(results, 15)


        return to_ret
            #if cor[i] > target_correlation:
            #    got_it = True
            #    print("cori", cor[i])
            #    results = np.append(results, [[cor[i], i]], axis=0)

            # print(self.quote.truncate(i, i+200, copy=True)['PRICE'].reset_index(drop=True))
            # print(cor[i])
        #max_index = cor.argmax()

        # best_lag = self.quote.truncate(max_index, max_index+lag_length, copy=True)['PRICE'].reset_index(drop=True)
        #print("The best matching: ", results)
        #print("max: ", np.abs(cor).max(), cor.argmax())
        #if got_it:
        #    return results
        #else:
        #    return 0


        #plt.plot(last_lag / last_lag.max() * 100)
        #for cor_i, cor_z in np.ndenumerate(cor):
        #    print(cor_i, cor_z)
        #    if cor_z > target_correlation:
        #        print(cor_i, cor_z)
        #        best_lag = alls.truncate(cor_i[0], cor_i[0] + lag_length * 2, copy=True).reset_index(
        #            drop=True)
        #        plt.plot(best_lag / best_lag.max() * 100)

        #plt.show()
        #return cor.max()



        #print(self.data.loc['2021-01-18 10:10:00'][col_name])

        #for stamp in range(0, len(self.data['DATETIME']):
        #   self.data['OTHER'].iloc(stamp) = tick.data[stamp]['CLOSE']

    def foreseeing_last(self, width):
        thebest = np.zeros((0, 2))

        print("lets find the oraculative lags", self.size)
        correlation_array = self.corr_price(self.size - width, width, self.target_correlation)

        print("the winner is:", correlation_array)
        self.plot_lag(self.size - width, width)
        for i in correlation_array:
            print(int(i[1]))
            self.plot_lag(i[1], width*2)

        plt.show()
    def whats_next(self):
        all_widths = np.zeros((0, 3))
        max_width = 3000
        step = 500
        search_depth = 5 #number of lines to get for each lag length

        for width in range(200, max_width-200, step):
            correlation_array = self.corr_price(self.size - width, width, self.target_correlation)
            all_widths = np.append(all_widths, get_max(correlation_array, search_depth), axis=0)
        print("all_width", all_widths)
        for x in range(0, all_widths.shape[0], search_depth):
            print('all widths', all_widths[x, 0])
            self.plot_lag(self.size - all_widths[x, 2], all_widths[x, 2])
            for y in range(0, search_depth):
                self.plot_lag(all_widths[x+y, 1], all_widths[x+y, 2]*2)
            plt.show()



    def looking_for_patterns(self):
        thebest = np.zeros((0, 2))

        print("lets find the best out of", sber.size)
        for i in range(0, sber.size - 200, 50):
            corre = np.size(sber.corr_price(i, 200, 0.98))
            thebest = np.append(thebest, [[corre, i]], axis=0)
            print("Processing lag:", i)
        print("here it is", thebest[thebest[:, 0].argsort()])
        sor = thebest[thebest[:, 0].argsort()]
        argmmaaxx = np.argmax(thebest[:, 0], axis=0)

        for index in range(1, 10):
            sber.plot_lag(sor[-index, 1], 200)
        plt.show()

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
                i=0
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



    def plot_lag(self, start, lag_length):
        print(start, lag_length)
        alls = self.slice['CLOSE'].reset_index(drop=True)
        lag = alls.truncate(start, start + lag_length, copy=True).reset_index(
            drop=True)
        print(lag)
        print("plot_lag call", self.slice.shape[0])
        plt.plot((lag-lag[0]) / lag.max() * 100)
        #plt.show()

    def plot_lag_padded(self, start, lag_length, max_width):
        print(start, lag_length)
        alls = self.slice['CLOSE'].reset_index(drop=True)
        lag = alls.truncate(start, start + lag_length, copy=True).reset_index(
            drop=True)
        #print("lag", lag)
        np_lag = np.zeros(lag.shape[0])
        np_lag = lag
        paddo = max_width - lag_length
        print("paddo", paddo)

        padded_lag = np.pad(np_lag, (int(paddo), 0), mode='constant')
        print(padded_lag)
        #print("plot_lag call", self.slice.shape[0])
        plt.plot((padded_lag - lag[0]) / lag.max() * 100)

    def plot_vectors(self, start, width, rgb=()):
        x = 0
        y = 0
        if not rgb:
            rgb = (random.random(), random.random(), random.random())
        vector = self.vector(start, width)

        for i in range(width):
            x1, y1 = [x, x + self.vectors[start + i, 0]], [y, y + self.vectors[start + i, 1]]
            xv, yv = [x, x + np.sign(self.vectors[start + i, 0]) * self.vectors[start + i, 2]*2], [y, y]
            x = x + self.vectors[start + i, 0]
            y = y + self.vectors[start + i, 1]
            plt.plot(y1, x1, color = rgb)
            plt.plot(yv, xv, color = rgb)

    def plot_nvectors(self, start, width, rgb=()):
        x = 0
        y = 0
        if not rgb:
            rgb = (random.random(), random.random(), random.random())
        vector = self.vector(start, width)

        for i in range(width):
            x1, y1 = [x, x + vector[i, 0]], [y, y + vector[i, 1]]
            xv, yv = [x, x + np.sign(vector[i, 0]) * vector[i, 2]], [y, y]
            x = x + vector[i, 0]
            y = y + vector[i, 1]
            plt.plot(y1, x1, color=rgb)
            plt.plot(yv, xv, color=rgb)




def cross_validation(sci_model, X, y):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error, r2_score, max_error
    print(f"Looking for best parameters...")

    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 20, 30, 50],
        "criterion": ["squared_error"],
    }

    gs = GridSearchCV(sci_model, param_grid, scoring="neg_root_mean_squared_error", cv=3)
    gs.fit(X, y)
    print("Best score" + str(gs.best_score_))
    print("Best params" + str(gs.best_params_))
    criterion1 = gs.best_params_['criterion']
    max_depth1 = gs.best_params_['max_depth']
    n_estimators1 = gs.best_params_['n_estimators']
    print(f"{criterion1}, {max_depth1}, {n_estimators1}")
    return gs.best_params_

def get_max(array, qty):

    sor = array[array[:, 0].argsort()]
    argmmaaxx = np.argmax(array[:, 0], axis=0)
    end = qty
    if np.size(sor)<qty:
        end = np.size(sor)

    return sor[-end:,:]

def corr_vectors(one, two, vector_correlation):
    yes=False

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
            yes=True
        else:
            return 0
    #print('return', y[0], x[0], leny, lenx)
    return 1

def normalize_list(vector):
    max = np.abs(vector).max()
    vector_norm = np.empty_like(vector)
    #print("max", max, vector)
    for i in range(0, len(vector)):
        vector_norm[i] = vector[i]/max*10
        #print(vector[i])
    #print(vector)
    return vector_norm

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

def load_vectors(self):
    with open('vectors.csv', 'a') as csvfile:
        vectors = np.loadtxt(csvfile, delimiter=",")
    return vectors








        #with open('out.csv', 'a', newline='', encoding='utf-8') as csvfile:
         #   csvfile.write(self.data['DATETIME'].astype(str)+' '+ self.data['CLOSE'].astype(str))
        #+' '+ self.data['CLOSE'].astype(str))

   # def dayvol(self):
    #    dv = pd.DataFrame()
     #   print(self.data['VOL'][date(2021, 9, 9)])
      #  dv['time'] = range(10, 19)
       # dv['vol'] = 0
        #for i in pd.date_range():



