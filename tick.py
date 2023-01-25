import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import date
from datetime import datetime
from dateutil.rrule import rrule, DAILY

#Basically creates a DataFrame with DATETIME, CLOSE and VOL columns based on file to be processed further on..
class Tick():
    def __init__(self, file):
        print("Creating Tick object...")
        names = ['TICKER','PER','DATE','TIME','CLOSE','VOL']
        self.data = pd.read_csv(file, delimiter=',', names=names)
        #print("Columns: " + self.data.columns)
        self.data['DATETIME'] = self.data['DATE'].astype(str) + self.data['TIME'].astype(str)
        self.data['DATETIME'] = pd.to_datetime(self.data['DATETIME'], format='%Y%m%d%H:%M')
        self.ticker = self.data.iloc[0]['TICKER']
        print('Extracted: ' + self.data.iloc[0]['TICKER'])
        self.data.drop(columns=['TICKER', 'PER', 'DATE', 'TIME'], inplace=True)

        #self.start = self.data['DATETIME'].iloc[0]
        #self.end = self.data['DATETIME'].iloc[-1]

        self.target_correlation = 0.85

        self.data.set_index('DATETIME', inplace=True)#, drop=False)
        self.size = self.data.shape[0]
        self.start = self.data.index[0]
        self.end = self.data.index[-1]
        print('Total entries here:', self.size)
        print(self.start, self.end)

        #self.meanvol = self.data.resample("M").mean()['VOL']
        #self.dayvol = self.data.groupby(self.data.index.minute).mean()['VOL']
        #self.meanclose = self.data.groupby(self.data.index.hour).mean()['CLOSE']
        #self.minclose = self.data.groupby(self.data.index.hour).min()['CLOSE']
        #self.maxvol = self.data.groupby(self.data.index.hour).max()['VOL']


        #print(self.data.index.freq)

        self.model = pd.DataFrame()
        self.to_model = pd.DataFrame()

        self.vol = np.array(self.data['VOL'])
        self.price = np.array(self.data['CLOSE'])
        self.vol = (self.vol - self.vol.min()) / (self.vol.max() - self.vol.min())
        self.mean_len = 0
        self.mean_price_move = 0


    #You can add only dataframe with DATETIME, CLOSE, VOLUME columns, DATETIME is index.
    def add(self, addition):
        self.data = pd.concat([self.data, addition])
        self.vol = np.array(self.data['VOL'])
        self.price = np.array(self.data['CLOSE'])
        self.vol = (self.vol - self.vol.min()) / (self.vol.max() - self.vol.min())
        self.size = self.data.shape[0]
        self.start = self.data.index[0]
        self.end = self.data.index[-1]


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
    def ema(self, len):
        alfa =  2 / (len + 1)
        ema = np.zeros(self.size)
        ema[0]=self.price[0]
        for j in range(1,len):
            ema[j]=np.average(self.price[:j])
        for i in range(len, self.size):
            ema[i]=alfa*self.price[i] + (1-alfa)*ema[i-1]
        return ema





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





    def plot_lag(self, start, lag_length, title='Lags'):
        print(start, lag_length)
        alls = self.slice['CLOSE'].reset_index(drop=True)
        lag = alls.truncate(start, start + lag_length, copy=True).reset_index(
            drop=True)
        print(lag)
        print("plot_lag call", self.slice.shape[0])
        plt.plot((lag-lag[0]) / lag.max() * 100)
        plt.title(title)
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
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error, r2_score, max_error
    print(f"Looking for best parameters...")

    #param_grid = {
    #    "n_estimators": [400],
    #    "max_depth": [300, 400],
    #    "criterion": ["squared_error"],
    #}
    param_grid = {"n_estimators": [500, 1000, 2000],
                  "max_depth": [7, 14, 21],
                  "eta": [0.1, 0.2, 0.3],
                  "subsample": [0.7, 0.8, 0.9],
                  "colsample_bytree": [0.7, 0.8, 0.9],
    }

    gs = GridSearchCV(sci_model, param_grid, scoring="neg_root_mean_squared_error", cv=3)
    gs.fit(X, y)
    print("Best score" + str(gs.best_score_))
    print("Best params" + str(gs.best_params_))

    #criterion1 = gs.best_params_['criterion']
    #max_depth1 = gs.best_params_['max_depth']
    n_estimators1 = gs.best_params_['n_estimators']
    max_depth1 = gs.best_params_['max_depth']
    eta1 = gs.best_params_['eta']
    subsample1 = gs.best_params_['subsample']
    colsample_bytree1 = gs.best_params_['colsample_bytree']

    #print(f"{criterion1}, {max_depth1}, {n_estimators1}")
    print(f"{eta1}, {max_depth1}, {n_estimators1}, {subsample1}, {colsample_bytree1}")
    return gs.best_params_

def get_max(array, qty):

    sor = array[array[:, 0].argsort()]
    argmmaaxx = np.argmax(array[:, 0], axis=0)
    end = qty
    if np.size(sor)<qty:
        end = np.size(sor)

    return sor[-end:,:]



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



