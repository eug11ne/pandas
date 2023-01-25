import numpy as np
from tick_vectors import Tick_vectors, plot_nvector_any
import pandas as pd
import math
import matplotlib.pyplot as plt

class Tick_model(Tick_vectors):
    def __init__(self, file, v_width, v_prominence):
        super().__init__(file, v_width, v_prominence)




    def fast_predict(self, model_name, lag_length=10):
        import joblib
        sci_model = joblib.load(model_name)
        X_p = self.vector_to_predict(lag_length)
        pred = sci_model.predict(X_p)
        return pred

    def vector_to_predict(self, lag_length=10):
        to_model = prep_xyv_tr_set(lag_length).columns
        last_lag = self.vectors.shape[0] - lag_length
        vector = self.vector(last_lag, lag_length)
        print("V to pred ", vector)
        X_p = pd.DataFrame(vector.reshape(1, -1), columns=to_model)
        return X_p

    def vector_to_predict_cos(self, lag_length=10):
        to_model = prep_xlcos_tr_set(lag_length)
        last_lag = self.vectors.shape[0] - lag_length
        #vector = self.vector(last_lag, lag_length)
        r_vector = self.vector(last_lag, lag_length)

        li = np.zeros(0)
        for j in range(lag_length):
            x = r_vector[j,0]
            l = math.sqrt(r_vector[j,0]**2+r_vector[j,1]**2)
            cos = x/l
            li = np.append(li, [x, l, cos])

        X_p = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
        print(X_p)
        return X_p




    def create_training_set(self, lag_length, steps=1):
        to_model = prep_xyv_tr_set(lag_length)
        print(to_model)
        for i in range(self.vectors.shape[0]-500):
            r_vector = self.vector(i, lag_length)
            li = np.zeros(0)
            for j in range(lag_length):
                x = r_vector[j, 0]
                y = r_vector[j, 1]
                v = r_vector[j, 0]/x #power needed to move 1 pt
                li = np.append(li, [x, y, v])

            temp = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
            #temp = pd.DataFrame(self.vector(i, lag_length).reshape(1, -1), columns=to_model.columns)
            to_model = pd.concat([to_model, temp], ignore_index=True)

        print("training_set", to_model)
        print("yay")
        column_numbers = [x for x in range(to_model.shape[1])]
        print(column_numbers)
        del column_numbers[-9:]
        print(column_numbers)
        col1 = f"X_{lag_length-1}"
        col2 = f"Y_{lag_length-1}"
        col3 = f"VOL_{lag_length-1}"
        #X = self.model.drop(columns=[col1, col2, col3])
        X = to_model.iloc[:,column_numbers]
        print(X)
        #y = pd.DataFrame()
        y1 = to_model.iloc[:,-9]
        y2 = to_model.iloc[:,-6]
        y3 = to_model.iloc[:,-3]
        y_y1 = to_model.iloc[:,-8]
        y_v1 = to_model.iloc[:,-7]
        print(y1, y2, y3)
        return X, y1, y2, y3, y_y1, y_v1

    def create_training_set_cos(self, lag_length, steps=1):
        to_model = prep_xlcos_tr_set(lag_length)
        print(to_model)
        for i in range(self.vectors.shape[0] - 500):
            r_vector = self.vector(i, lag_length)
            li = np.zeros(0)
            for j in range(lag_length):
                x = r_vector[j, 0]
                l = math.sqrt(r_vector[j, 0] ** 2 + r_vector[j, 1] ** 2)
                cos = x / l
                li = np.append(li, [x, l, cos])

            temp = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
            to_model = pd.concat([to_model, temp], ignore_index=True)

        print("training_set", to_model)
        print("yay")
        column_numbers = [x for x in range(to_model.shape[1])]
        print(column_numbers)
        del column_numbers[-9:]
        print(column_numbers)
        col1 = f"X_{lag_length-1}"
        col2 = f"Y_{lag_length-1}"
        col3 = f"VOL_{lag_length-1}"
        #X = self.model.drop(columns=[col1, col2, col3])
        X = to_model.iloc[:,column_numbers]
        print(X)
        #y = pd.DataFrame()
        y1 = to_model.iloc[:,-9]
        y2 = to_model.iloc[:,-6]
        y3 = to_model.iloc[:,-3]
        y_y1 = to_model.iloc[:,-8]
        y_v1 = to_model.iloc[:,-7]
        print(y1, y2, y3)
        return X, y1, y2, y3, y_y1, y_v1

    def predict_vectors(self, vector, pred_len):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        import joblib

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
        train_len = 17000
        test_len = -2000
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
        joblib.dump(sci_model, "model-1step.pkl")
        y_pred1 = sci_model.predict(X_p)
        # y_pred = sci_model.predict(X)
        print("******model")
        print(X_p)
        print("Model score:", sci_model.score(X_test, y1_test))
        print("Prediction for 1 steps ahead", pred_len, y_pred1)
        sci_model.fit(X_train, y2_train)
        joblib.dump(sci_model, "model-2step.pkl")
        y_pred2 = sci_model.predict(X_p)
        print("Model score:", sci_model.score(X_test, y2_test))
        print("Prediction for 2 steps ahead", pred_len, y_pred2)
        sci_model.fit(X_train, y3_train)
        joblib.dump(sci_model, "model-3step.pkl")
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

    def create_model_for_vectors(self, v_length=10):
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        print(f"Modelling vectors for {self.data.iloc[1]['TICKER']}...")

        X, y1, y2, y3, y_y1, y_v1 = self.create_training_set(v_length+3)

        train_len = self.vectors.shape[0]-500-int(self.vectors.shape[0]/10)
        test_len = -(self.vectors.shape[0]+500)
        X_train = X[:train_len]
        y1_train = y1[:train_len]
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

        sci_model = RandomForestRegressor(n_estimators=200, max_depth=v_length, criterion='squared_error')
        # sci_model = RandomForestRegressor(n_estimators=n_estimators1, max_depth=max_depth1, criterion=criterion1)
        #cross_validation(sci_model, X_train, y1_train)
        sci_model.fit(X_train, y1_train)
        joblib.dump(sci_model, f"model-1step_{v_length}.pkl")

        print("******model")
        print("Model score:", sci_model.score(X_test, y1_test))

        sci_model.fit(X_train, y2_train)
        joblib.dump(sci_model, f"model-2step_{v_length}.pkl")

        print("Model score:", sci_model.score(X_test, y2_test))

        sci_model.fit(X_train, y3_train)
        joblib.dump(sci_model, f"model-3step_{v_length}.pkl")

        print("Model score:", sci_model.score(X_test, y3_test))

        #sci_model.fit(X_train, y_y1_train)
        #joblib.dump(sci_model, f"model-1step-y1_{v_length}.pkl")
        #print("Model score:", sci_model.score(X_test, y_y1_test))


        #sci_model.fit(X_train, y_v1_train)
        #joblib.dump(sci_model, f"model-1step-v1_{v_length}.pkl")
        #print("Model score:", sci_model.score(X_test, y_v1_test))

        # plt.plot(y_pred)
        print("end model")

        return 0

def prep_xlcos_tr_set(lag_length):
    to_model = pd.DataFrame()
    for j in range(lag_length):
        to_model[f'X_{j}'] = []
        to_model[f'L_{j}'] = []
        to_model[f'COS_{j}'] = []

    return to_model


def prep_xyv_tr_set(lag_length):
    to_model = pd.DataFrame()

    for j in range(lag_length):
        to_model[f'X_{j}'] = []
        to_model[f'Y_{j}'] = []
        to_model[f'VOL_{j}'] = []
    return to_model



