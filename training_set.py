from tick_vectors import Tick_vectors, normalize_list
from tick_model import Tick_model
import numpy as np
import pandas as pd
from tick import cross_validation
import math

class Tick_training_set():
    def __init__(self, vectors_file, lag_length, type):
        self.length = lag_length + 3
        self.type = type
        #self.features = features
        with open(vectors_file) as csvfile:
            self.vectors = np.loadtxt(csvfile, delimiter=",")
        print("Vectors loaded: ", self.vectors.shape[0])
        self.create(self.type)

    def create(self, type):
        if type == 'sum':
            self.X, self.y1, self.y2, self.y3, self.y_y1, self.y_v1 = self.create_sum()
        elif type == 'sum2':
            self.X, self.y1, self.y2, self.y3, self.y_y1, self.y_v1 = self.create_sum2()
        elif type == 'cos':
            return 0
        elif type == 'std':
            self.X, self.y1, self.y2, self.y3, self.y_y1, self.y_v1 = self.create_std()
        elif type == 'sum2v':
            self.X, self.y1, self.y2, self.y3, self.y_y1, self.y_v1 = self.create_sum2v()



    def vector(self, start, width):

        vector = self.vectors[start:start+width,:]
        norm_vector = np.empty_like(vector)
        norm_vector[:, 0] = normalize_list(vector[:, 0]) #x
        norm_vector[:, 1] = normalize_list(vector[:, 1]) #y
        norm_vector[:, 2] = normalize_list(vector[:, 2]) #volume
        norm_vector[:, 3] = normalize_list(vector[:, 3]) #ema1
        norm_vector[:, 4] = normalize_list(vector[:, 4]) #ema2
        return norm_vector

    def create_sum(self):
        tr_columns = ['X', 'SUM', 'Y']
        to_model = self.prep(tr_columns)

        print(to_model)
        for i in range(self.length, self.vectors.shape[0] - self.length):
            r_vector = self.vector(i-self.length, self.length*2)
            li = np.zeros(0)

            for j in range(self.length):
                sum = 0
                x = r_vector[self.length+j, 0]
                for k in range(self.length):
                    sum += r_vector[j+k, 0]

                y = r_vector[self.length+j, 1]

                li = np.append(li, [x, sum, y])

            temp = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
            to_model = pd.concat([to_model, temp], ignore_index=True)

        return cut_tr_set(to_model, features=3)

    def create_sum2(self):
        tr_columns = ['X', 'SUM1', 'SUM2', 'Y']
        to_model = self.prep(tr_columns)

        print(to_model)
        for i in range(self.length*2, self.vectors.shape[0] - self.length):
            r_vector = self.vector(i-self.length*2, self.length*3)
            li = np.zeros(0)

            for j in range(self.length):
                sum = 0
                summ = 0
                x = r_vector[self.length*2+j, 0]
                for k in range(self.length):
                    sum += r_vector[j+self.length+k, 0]
                    #print(sum, k, r_vector[j-self.length+k, 0])
                #print('**** ', sum)

                for k2 in range(self.length*2):
                    summ += r_vector[j + k2, 0]
                y = r_vector[self.length*2+j, 1]
                li = np.append(li, [x, sum, summ, y])

            temp = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
            # temp = pd.DataFrame(self.vector(i, lag_length).reshape(1, -1), columns=to_model.columns)
            to_model = pd.concat([to_model, temp], ignore_index=True)
        return cut_tr_set(to_model, features=4)

    def create_std(self):
        tr_columns = ['X', 'LEN', 'EMA1', 'EMA2', 'EMADIFF',  'Y', 'VOL', 'RATIO']
        features = len(tr_columns)
        to_model = self.prep(tr_columns)

        print(to_model)
        for i in range(self.length, self.vectors.shape[0] - self.length):
            r_vector = self.vector(i-self.length, self.length)
            li = np.zeros(0)


            for j in range(self.length):

                x = r_vector[j, 0]
                x_prev = r_vector[j - 1, 0] if j>0 else 1

                y = r_vector[j, 1]
                ratio = x/abs(x_prev)
                vol = r_vector[j, 2]
                length = np.sqrt(x**2 + y**2)
                ema1 = r_vector[j, 3]
                ema2 = r_vector[j, 4]
                emadiff = ema1 - ema2
                #ratio = x/y #abs(x_prev)

                li = np.append(li, [x, length, ema1, ema2, emadiff, y, vol, ratio])

            temp = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
            to_model = pd.concat([to_model, temp], ignore_index=True)

        return cut_tr_set(to_model, features=features)

    def create_sum2v(self):
        tr_columns = ['X', 'SUM1', 'SUM2', 'POWER', 'LEN', 'RATIO', 'Y']
        to_model = self.prep(tr_columns)

        print(to_model)
        for i in range(self.length*2+1, self.vectors.shape[0] - self.length):
            r_vector = self.vector(i-self.length*2, self.length*3)
            li = np.zeros(0)

            for j in range(self.length):
                sum = 0
                summ = 0
                x = r_vector[self.length*2+j, 0]
                x_prev = r_vector[self.length*2+j-1, 0]
                for k in range(self.length):
                    sum += r_vector[j+self.length+k, 0]
                    #print(sum, k, r_vector[j-self.length+k, 0])
                #print('**** ', sum)

                for k2 in range(self.length*2):
                    summ += r_vector[j + k2, 0]

                y_prev = r_vector[self.length*2+j-1, 1]
                power = r_vector[self.length*2 + j, 2] #/ r_vector[self.length + j, 0]
                y = 1 if power > 7 else 0
                vlen = math.sqrt(x**2 + y**2)
                vratio = x/abs(x_prev)
                li = np.append(li, [x, sum, summ, power, vlen, vratio, y])

            temp = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
            # temp = pd.DataFrame(self.vector(i, lag_length).reshape(1, -1), columns=to_model.columns)
            to_model = pd.concat([to_model, temp], ignore_index=True)
        return cut_tr_set(to_model, features=7)


    def prep(self, features):

        names = []

        for j in range(self.length):
            for f in features:
                names.append(f'{f}_{j}')
                #to_model[f'{f}_{j}'] = []
        to_model = pd.DataFrame(columns=names)
        return to_model

    def create_model(self):
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        import joblib

        print(f"Modelling vectors...")

        #X, y1, y2, y3, y_y1, y_v1 = self.create_training_set(v_length+3)

        train_len = self.vectors.shape[0]-int(self.vectors.shape[0]/10)
        test_len = -(int(self.vectors.shape[0]/10))
        X_train = self.X[:train_len]
        y1_train = self.y1[:train_len]
        y2_train = self.y2[:train_len]
        y3_train = self.y3[:train_len]
        y_y1_train = self.y_y1[:train_len]
        y_v1_train = self.y_v1[:train_len]
        X_test = self.X[test_len:]
        y1_test = self.y1[test_len:]
        y2_test = self.y2[test_len:]
        y3_test = self.y3[test_len:]
        y_y1_test = self.y_y1[test_len:]
        y_v1_test = self.y_v1[test_len:]

        #sci_model = RandomForestRegressor(n_estimators=200, max_depth=self.length, criterion='squared_error')
        #model_type = 'rf'
        model_type = 'gb'
        sci_model = XGBRegressor(n_estimators=500, max_depth=14, eta=0.1, subsample=0.9, colsample_bytree=0.7)
        # sci_model = RandomForestRegressor(n_estimators=n_estimators1, max_depth=max_depth1, criterion=criterion1)
        #cross_validation(sci_model, X_train, y1_train)

        #sci_model.fit(X_train, y_y1_train)
        # joblib.dump(sci_model, f"model-1step-y1_{v_length}.pkl")
        #print("Model score:", sci_model.score(X_test, y_y1_test))

        sci_model.fit(X_train, y1_train)
        joblib.dump(sci_model, f"{model_type}_model-1step_{self.length}_{self.type}.pkl")

        print("******model")
        print("Model score:", sci_model.score(X_test, y1_test))

        sci_model.fit(X_train, y2_train)
        joblib.dump(sci_model, f"{model_type}_model-2step_{self.length}_{self.type}.pkl")

        print("Model score:", sci_model.score(X_test, y2_test))

        sci_model.fit(X_train, y3_train)
        joblib.dump(sci_model, f"{model_type}_model-3step_{self.length}_{self.type}.pkl")

        print("Model score:", sci_model.score(X_test, y3_test))




        #sci_model.fit(X_train, y_v1_train)
        #joblib.dump(sci_model, f"model-1step-v1_{v_length}.pkl")
        #print("Model score:", sci_model.score(X_test, y_v1_test))

        # plt.plot(y_pred)
        print("end model")

        return 0

    def find_best_model(self):
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        import joblib

        print(f"Modelling vectors...")

        # X, y1, y2, y3, y_y1, y_v1 = self.create_training_set(v_length+3)

        train_len = self.vectors.shape[0] - int(self.vectors.shape[0] / 10)
        test_len = -(int(self.vectors.shape[0] / 10))
        X_train = self.X[:train_len]
        y1_train = self.y1[:train_len]
        y2_train = self.y2[:train_len]
        y3_train = self.y3[:train_len]
        y_y1_train = self.y_y1[:train_len]
        y_v1_train = self.y_v1[:train_len]
        X_test = self.X[test_len:]
        y1_test = self.y1[test_len:]
        y2_test = self.y2[test_len:]
        y3_test = self.y3[test_len:]
        y_y1_test = self.y_y1[test_len:]
        y_v1_test = self.y_v1[test_len:]

        #sci_model = RandomForestRegressor(n_estimators=400, max_depth=300, criterion='squared_error')
        sci_model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
        cross_validation(sci_model, X_train, y1_train)

def cut_tr_set(to_model, features=3):
    print("training_set", to_model)
    print("yay")
    column_numbers = [x for x in range(to_model.shape[1])]
    print(column_numbers)
    del column_numbers[-features*3:]
    print(column_numbers)
    X = to_model.iloc[:, column_numbers]
    print(X)
    y1 = to_model.iloc[:, -features*3]
    print(y1)
    y2 = to_model.iloc[:, -features*2]
    print(y2)
    y3 = to_model.iloc[:, -features]
    print(y3)
    y_y1 = to_model.iloc[:, -features*3+1]
    y_v1 = to_model.iloc[:, -features*3+2]
    #print(y1, y2, y3)
    return X, y1, y2, y3, y_y1, y_v1













