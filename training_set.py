from tick_vectors import Tick_vectors, normalize_list
from tick_model import Tick_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tick import cross_validation
import math

class Tick_training_set():
    def __init__(self, vectors_file, lag_length, type, steps=3, nn=(3,3), columns=['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'TOEMA1', 'TOEMA2']):
        self.length = lag_length + 3
        self.vectors_file = vectors_file
        self.nn=nn
        self.steps_ahead = steps
        self.lag_length = lag_length
        self.type = type
        self.features_number = 0
        self.epochs_number = 200
        self.batch_size = 500
        self.tr_columns = columns#['X', 'TIME', 'LEN', 'RATIO', 'SIN', 'TOEMA1', 'TOEMA2', 'EMADIFF', 'EMA1', 'EMA2', 'DIFF1', 'DIFF2']#, 'DDIFF1', 'DDIFF2']#'TOEMA1', 'TOEMA2']#, 'LEN', 'VOLX']#, 'DIFF2', 'DIFF1', 'DDIFF1', 'DDIFF2']#, 'EMADIFF']#, , 'VOLX', 'TOEMA1', 'TOEMA2']# 'EMADIFF', 'TOEMA1', 'TOEMA2']#, 'RATIO', 'VOLX', 'SIN']  # , 'EMA1', 'EMA2', 'DIFF1', 'DIFF2', 'DDIFF1', 'DDIFF2']
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
            self.create_std()
        elif type == 'simple':
            self.create_simple()
        elif type == 'lstm':
            self.create_lstm()
        elif type == 'all':
            self.create_whole_lstm()
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
        norm_vector[:, 5] = vector[:, 5]#sin
        norm_vector[:, 6] = vector[:, 6]#diff1
        norm_vector[:, 7] = vector[:, 7]#diff2
        norm_vector[:, 8] = vector[:, 8]#2nd diff2
        norm_vector[:, 9] = vector[:, 9]#2nd diff2
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
        to_model_minus = self.prep(tr_columns)


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
            if temp[f'X_{self.length-4}'][0]<0:
                to_model = pd.concat([to_model, temp], ignore_index=True)
            else:
                to_model_minus = pd.concat([to_model_minus, temp], ignore_index=True)
        self.X, self.y1, self.y2, self.y3, self.y_y1, self.y_v1 = cut_tr_set(to_model, features=features)
        self.X_minus, self.y1_minus, self.y2_minus, self.y3_minus, self.y_y1_minus, self.y_v1_minus = cut_tr_set(to_model_minus, features=features)

        return True

    def create_simple(self):
        into_future = self.steps_ahead
        self.length = self.lag_length + into_future
        features = len(self.tr_columns)
        to_model = self.prep(self.tr_columns)
        #to_model_minus = self.prep(self.tr_columns)

        print(to_model)
        for i in range(self.length+1500, self.vectors.shape[0] - self.length):
            r_vector = self.vector(i-self.length, self.length+1) #changed
            #li = np.zeros(0)
            li = simple_prep(r_vector, self.length, self.tr_columns)

            temp = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
            to_model = pd.concat([to_model, temp], ignore_index=True)

            #if temp[f'X_{self.length-4}'][0]<0:
            #    to_model = pd.concat([to_model, temp], ignore_index=True)
            #else:
            #    to_model_minus = pd.concat([to_model_minus, temp], ignore_index=True)

        self.X, self.y1, self.y2, self.y3, self.y_y1, self.y_v1 = cut_tr_set_long(to_model, into_future, features=features)
        print("CHESCKING ISNAN")
        if np.any(np.isnan(self.X)):
            print('IS NAN IN X')
        if np.any(np.isnan(self.y1)):
            print('IS NAN IN Y')
        #self.X_minus, self.y1_minus, self.y2_minus, self.y3_minus, self.y_y1_minus, self.y_v1_minus = cut_tr_set(to_model_minus, features=features)

        return True
    def create_whole_lstm(self):
        from sklearn.preprocessing import MinMaxScaler
        train_data_x = []
        train_data_x2 = []
        train_data_x3 = []
        train_data_y = []
        test=300

        steps = self.length-3

        for i in range(self.length, self.vectors.shape[0] - self.length):
            vec = self.vector(i-self.length, self.length+1)
            #temp = self.vectors[i:i+steps,0]
            temp = vec[:-2,0]
            temp1 = vec[:-2,1]
            train_data_x.append(temp.T)
            temp = self.vectors[i:i+steps,1]
            #train_data_x2.append(temp1.T)
            #temp = self.vectors[i:i+steps,2]
            #train_data_x3.append(temp.T)

            #y = self.vectors[i+steps+1,0]
            y = vec[-1,0]#yvec = self.vectors[i+steps+1,0]
            train_data_y.append(y)

        #print('temp', temp)
        #print('tempT', temp.T)


        x_train, y_train = np.array(train_data_x), np.array(train_data_y)
        x_train2, x_train3 = np.array(train_data_x2), np.array(train_data_x3)
        print(x_train.shape)
        #print(y_train.shape)
        print('X train', x_train)
        #print('X train0', x_train[:,:,0])
        #print(y_train)
        #sc = MinMaxScaler(feature_range=(0, 1))
        #x_train_scaled = sc.fit_transform(x_train)
        self.x=x_train[:-300]
        self.xt=x_train[-301:]
        self.x2 = x_train2[:-300]
        self.x2t = x_train2[-301:]
        # self.xt=x_train_scaled[-301:,:]
        #self.x=x_train_scaled[:-300,:]
        #self.xt=x_train_scaled[-301:,:]
        #self.scaler = MinMaxScaler(feature_range=(0, 1))

        #y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1,1))
        self.y = y_train[:-300].T
        self.yt = y_train[-301:].T
        #self.y = y_train_scaled[:-300, :]
        #self.yt = y_train_scaled[-301:, :]
        print(self.x.shape)
        print(self.x)
        print(self.y, 'Y')
        #answer = y_train.reshape(-1,1)
        #self.scaler.fit(answer)


        return True


    def create_lstm(self):
        from sklearn.preprocessing import MinMaxScaler
        tr_columns = ['X', 'RATIO']#'EMA1', 'EMA2', 'RATIO']
        features = len(tr_columns)
        to_model = self.prep_lstm(tr_columns)


        print(to_model)
        for i in range(self.length, self.vectors.shape[0] - self.length):
            r_vector = self.vector(i-self.length, self.length)
            li = np.zeros(0)

            for j in range(self.length):
                x = r_vector[j, 0]
                #sign = np.sign(r_vector[j, 0])

                li = np.append(li, [x])

            for k in range(self.length):
                ema1 = r_vector[k, 2]
               #sign = np.sign(r_vector[j, 0])

                li = np.append(li, [ema1])

            #for k in range(self.length):

            #    ema2 = r_vector[k, 4]
            #    sign = np.sign(r_vector[k, 0])
            #    li = np.append(li, [sign])

            #for k in range(self.length):
            #   x_prev = r_vector[k - 1, 0] if k > 0 else 1
            #   ratio = x_prev/r_vector[k, 0]
            #   sign = np.sign(r_vector[k, 1])

            #   li = np.append(li, [ratio])

            temp = pd.DataFrame(li.reshape(1, -1), columns=to_model.columns)
            #if temp[f'X_{self.length - 4}'][0] < 0:
            to_model = pd.concat([to_model, temp], ignore_index=True)

        sc = MinMaxScaler(feature_range=(0, 1))
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        answer = to_model.iloc[:, self.length-3].to_numpy().reshape(-1,1)
        self.scaler.fit(answer)

        to_model = pd.DataFrame(sc.fit_transform(to_model), columns=to_model.columns)

        self.X, self.y1, self.y2, self.y3, self.y_y1, self.y_v1 = cut_tr_set_lstm(to_model, features=features)


        return True



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
        self.features_number = len(features)

        for j in range(self.length):
            for f in features:
                names.append(f'{f}_{j}')
                #to_model[f'{f}_{j}'] = []
        to_model = pd.DataFrame(columns=names)
        return to_model

    def prep_lstm(self, features):

        names = []
        self.features_number = len(features)

        for f in features:
            for j in range(self.length):
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

        #sci_model.fit(X_train, y_y1_train)
        #joblib.dump(sci_model, f"{model_type}_model-time_{self.length}_{self.type}.pkl")

        #print("Model score:", sci_model.score(X_test, y_y1_test))


        print("end model")

        return 0

    def create_nn(self, model_number=1, mins=5, lstm=True):

        X_train, price1_train, price2_train, price3_train, X_test, price1_test, price2_test, price3_test = self.split_training_set(plus=True)
        #X_train_m, price1_train_m, price2_train_m, price3_train_m, X_test_m, price1_test_m, price2_test_m, price3_test_m = self.split_training_set(plus=False)
        print("********************************************")
        print(X_train)
        print(price1_train)


        #self.train_nn(self.x, self.y, self.xt, self.yt, 1, plus=True, model_number=model_number, lstm=False)
        preds, ytest = self.train_nn(X_train, price1_train, X_test, price1_test, 1, plus=True, model_number=model_number, mins=mins, lstm=False)
        #self.train_nn(X_train_m, price1_train_m, X_test_m, price1_test_m, 1, plus=False, model_number=model_number, lstm=lstm)
        #self.train_nn(X_train, price2_train, X_test, price2_test, 2, plus=True, model_number=model_number, lstm=lstm)
        #self.train_nn(X_train_m, price2_train_m, X_test_m, price2_test_m, 2, plus=False, model_number=model_number, lstm=lstm)
        #self.train_nn(X_train, price3_train, X_test, price3_test, 3, plus=True, model_number=model_number, lstm=lstm)
        #self.train_nn(X_train_m, price3_train_m, X_test_m, price3_test_m, 3, plus=False, model_number=model_number, lstm=lstm)

        #self.train_nn(X_train, price2_train, X_test, price2_test, 2)
        #self.train_nn(X_train, price3_train, X_test, price3_test, 3)
        return preds, ytest


    def train_nn(self, X, Y, X_test, Y_test, step, plus, lstm=False, model_number=1, mins=5):
        from keras.callbacks import EarlyStopping
        from keras.models import Sequential
        from keras.optimizers import Adam
        from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, BatchNormalization
        from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
        from sklearn.preprocessing import MinMaxScaler

        model = Sequential()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', restore_best_weights=True)
        if lstm:

            #Xn=X.to_numpy().reshape(X.shape[0], self.length-3, self.features_number)
            #Xn=X.to_numpy().reshape(X.shape[0], self.length-3, 2)
            Xn=X.reshape(X.shape[0], self.length-3, 1)
            X_testn =X_test.reshape(X_test.shape[0], self.length - 3, 1)
            print(Xn[0,:,:])
            #X_testn=X_test.to_numpy().reshape(X_test.shape[0], self.length-3, self.features_number)
            #X_testn=X_test.to_numpy().reshape(X_test.shape[0], self.length-3, 2)
            #Yn=Y.to_numpy().reshape(int(Y.shape[0]/5), self.length-3)
            #Y_testn=Y_test.to_numpy().reshape(int(Y_test.shape[0]/5), self.length-3)
            print("NUMBER OF string, columns", X.shape)
            #print("NUMBER OF string, columns, test", X_testn.shape)
            #self.length-3, self.features_number
            model.add(LSTM(50, input_shape=(self.length-3, 1), activation="relu", return_sequences=True))
            #model.add(LSTM(20, batch_input_shape=(1, 1, self.length-3), activation="relu", stateful=True, return_sequences=False))
            #model.add(LSTM(64, batch_input_shape=(50, self.length-3, self.features_number), activation="relu", stateful=True, return_sequences=True))
            #model.add(Dense(X.shape[1], activation="relu"))
            #model.add(Dropout(0.2))
            #model.add(LSTM(50, activation="relu", return_sequences=True))
            #model.add(Dropout(0.2))
            #model.add(LSTM(50, activation="relu", return_sequences=True))
            #model.add(Dropout(0.2))
            #model.add(Bidirectional(LSTM(50, return_sequences=True)))

            #model.add(Dropout(0.2))
            model.add(LSTM(50, activation="relu", return_sequences=False))
            #model.add(Bidirectional(LSTM(30, return_sequences=False)))
            #model.add(Dense(X.shape[1]*3, activation="relu"))
            model.add(Dropout(0.2))
            #model.add(LSTM(64, activation="relu", return_sequences=True))


            #model.add(Flatten())

            model.add(Dense(50, activation="relu"))
            model.add(Dropout(0.2))
            #model.add(Dense(X.shape[1]*2, activation="relu"))
            #model.add(Dropout(0.2))
            model.add(Dense(X.shape[1], activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            #model.add(Dropout(0.2))
            #opt = Adam(learning_rate=0.01)
            model.compile(optimizer='adam', loss="mse", metrics='accuracy') #'accuracy'
            model.summary()
            #model.fit(x=Xn, y=Y, batch_size=10, epochs=50, validation_data=(X_testn, Y_test), callbacks=[early_stopping])
            model.fit(x=Xn, y=Y, batch_size=10, epochs=20, validation_data=(X_testn, Y_test), callbacks=[early_stopping])
            #model.compile(optimizer="adam", loss="mse", metrics='mean_absolute_error')
        else:
            #model.add(LeakyReLU(alpha=0.05))
            Xn=X
            X_testn = X_test
            drop_out = 0.2
            print('test', X_testn.shape)
            model.add(Dense(X.shape[1], input_shape=(X.shape[1],), activation="relu")) #*X.shape[1]

            model.add(Dropout(drop_out))
            layers_number = self.nn[0]
            layers_multiplicator = self.nn[1]
            for layer in range(1,layers_number+1):
                #model.add(Dense(X.shape[1] * layer * layers_multiplicator, activation="relu"))
                model.add(Dense(X.shape[1] * layers_multiplicator, activation="relu"))
                model.add(Dropout(drop_out))
            #model.add(BatchNormalization)
            for layer in range(layers_number-1,0,-1):
                #model.add(Dense(X.shape[1] * layer * layers_multiplicator, activation="relu"))
                model.add(Dense(X.shape[1] * layers_multiplicator, activation="relu"))
                model.add(Dropout(drop_out))
            '''
            
            model.add(Dense(X.shape[1]*2, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(X.shape[1]*3, activation="relu"))
            model.add(Dropout(0.2))            
            model.add(Dense(X.shape[1] * 3*2, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(X.shape[1] * 3*3, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(X.shape[1] * 3*2, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(X.shape[1] * 3, activation="relu"))
            model.add(Dropout(0.2))
            #model.add(Dense(X.shape[1] * 3, activation="relu"))
            #model.add(Dropout(0.2))
            #model.add(Dense(X.shape[1] * 3, activation="relu"))
            #model.add(Dropout(0.2))
            #model.add(Dense(X.shape[1] * 3, activation="relu"))
            #model.add(Dropout(0.2))
            #model.add(Dense(X.shape[1] * 3, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(X.shape[1] * 2, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(X.shape[1] * 2, activation="relu"))
            model.add(Dropout(0.2))'''
            model.add(Dense(X.shape[1], activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            #model.build()#optimizer="adam", loss="mse", metrics='mean_absolute_error')
            model.compile(optimizer="adam", loss="mse", metrics='mean_absolute_error')
            model.summary()
            #history = model.fit([X_train1, X_train2], y_train, validation_data=([X_test1, X_test2], y_test),
             #                   epochs=1000, batch_size=32)
            model.fit(x=Xn, y=Y, batch_size=self.batch_size, epochs=self.epochs_number, validation_data=(X_testn, Y_test))#, callbacks=[early_stopping])


        predictions = model.predict(X_testn)
        #sc2=MinMaxScaler(feature_range=(0,1))
        #z = sc2.fit_transform(Y_test.to_numpy().reshape(-1,1))
        #predictions = self.scaler.inverse_transform(predictions.reshape(-1,1))
        #z= self.scaler.inverse_transform(Y_test.to_numpy().reshape(-1,1))
        #z= self.scaler.inverse_transform(Y_test.reshape(-1,1))
        #checko = sc2.inverse_transform(Y_test).T
        right_direction=0
        wrong_direction=0
        good_guess=0
        norm_guess=0
        big_guess=0
        small_guess=0
        total_bigs = 0
        total_smalls=0
        for i, j in zip(predictions, Y_test):#Y_test.to_numpy().T):
            print(f"predicted: {i}, actual: {j}")
            if i*j>0:
                right_direction+=1
            else:
                wrong_direction+=1
            if abs(j-i) < 0.1:
                good_guess+=1
            elif abs(j-i) < 0.2:
                norm_guess+=1
        for i, j in zip(predictions, Y_test):
            if abs(i) > 0.85 or abs(j) > 0.85:
                print(f"predicted big: {i}, actual big: {j}")
                if abs(i) > 0.7 and abs(j) > 0.7:
                    big_guess+=1
                if abs(j) > 0.7:
                    total_bigs+=1
            if abs(i) < 0.1 or abs(j) < 0.1:
                print(f"predicted small: {i}, actual small: {j}")
                if abs(i) < 0.19 and abs(j) < 0.1:
                    small_guess += 1
                if abs(j) < 0.1:
                    total_smalls += 1
        print(f"Lag length: {self.lag_length}, step: {step}, NN size: {self.nn}, dalaset size: {len(self.vectors)}")
        print(f"Good ones: {good_guess}, ok ones: {norm_guess} ({(good_guess+norm_guess)/(right_direction+wrong_direction)*100}%), good bigs: {big_guess} of {total_bigs} ({big_guess/total_bigs*100}%), good smalls: {small_guess} of {total_smalls} ({small_guess/total_smalls*100}%)\n")
        print(f"Right direction guess: {right_direction}, wrong direction guess: {wrong_direction}, coefficient: {right_direction/wrong_direction}")
        with open('stats.txt', 'a') as fl:
            #for i, j in zip(predictions, Y_test):
                #if abs(i) > 0.85 or abs(j) > 0.85:
                    #fl.write(f"predicted big: {i}, actual big: {j}\n")
            fl.write(f"Training data file: {self.vectors_file}\n")
            fl.write(f"Lag length: {self.length}, step: {step}, NN size: {self.nn}, dalaset size: {len(self.vectors)}\n")
            fl.write(f"Good ones: {good_guess}, ok ones: {norm_guess} ({(good_guess+norm_guess)/(right_direction+wrong_direction)*100}%), good bigs: {big_guess} of {total_bigs} ({big_guess / total_bigs * 100}%), good smalls: {small_guess} of {total_smalls} ({small_guess / total_smalls * 100}%)\n")
            fl.write(f"Right direction guess: {right_direction}, wrong direction guess: {wrong_direction}, coefficient: {right_direction / wrong_direction}\n")
            fl.write(f"The absolute mean error: {mean_absolute_error(Y_test, predictions)}\n")
            fl.write(f"The squared mean error: {mean_squared_error(Y_test, predictions)}\n")
            fl.write(f"Sqrt of the squared mean error: {np.sqrt(mean_squared_error(Y_test, predictions))}\n")
            fl.write(f"Columns: {self.tr_columns}\n\n")

        #print(predictions)
        #print("actual values")
        #print(Y_test.to_numpy().T)
        print("The absolute mean error :", mean_absolute_error(Y_test, predictions))
        print("The squared mean error :", mean_squared_error(Y_test, predictions))
        print("Sqrt of the squared mean error :", np.sqrt(mean_squared_error(Y_test, predictions)))

        forbar = [i[0] for i in predictions]
        y_forbar = [y for y in Y_test.to_numpy()]
        print(forbar)
        print(y_forbar)
        #plt.plot(forbar)
        #plt.plot(Y_test.to_numpy().T)
        #plt.show()
        #print(predictions)

        model.save(f"keras_{step}step_{self.length-3}_mins{mins}_{model_number}.nnn", save_format="h5")

        return forbar, y_forbar


    def split_training_set(self, plus=True):
        plus=True#divider=int((self.length - 3)**2)
        if plus:
            train_len = self.X.shape[0] - int(self.X.shape[0] / 10)
            test_len = -(int(self.X.shape[0] / 10))
            X_train = self.X[:train_len]
            price1_train = self.y1[:train_len]
            price2_train = self.y2[:train_len]
            price3_train = self.y3[:train_len]
            y_y1_train = self.y_y1[:train_len]
            y_v1_train = self.y_v1[:train_len]
            X_test = self.X[test_len:]
            price1_test = self.y1[test_len:]
            price2_test = self.y2[test_len:]
            price3_test = self.y3[test_len:]
            y_y1_test = self.y_y1[test_len:]
            y_v1_test = self.y_v1[test_len:]
        else:
            train_len = self.X_minus.shape[0] - int(self.X_minus.shape[0] / 20)
            test_len = -(int(self.X_minus.shape[0] / 20))
            X_train = self.X_minus[:train_len]
            price1_train = self.y1_minus[:train_len]
            price2_train = self.y2_minus[:train_len]
            price3_train = self.y3_minus[:train_len]
            y_y1_train = self.y_y1_minus[:train_len]
            y_v1_train = self.y_v1_minus[:train_len]
            X_test = self.X_minus[test_len:]
            price1_test = self.y1_minus[test_len:]
            price2_test = self.y2_minus[test_len:]
            price3_test = self.y3_minus[test_len:]
            y_y1_test = self.y_y1_minus[test_len:]
            y_v1_test = self.y_v1_minus[test_len:]
        return X_train, price1_train, price2_train, price3_train, X_test, price1_test, price2_test, price3_test

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

def cut_tr_set_long(to_model, lag_length, features=3):
    print("training_set", to_model)
    print("yay")
    column_numbers = [x for x in range(to_model.shape[1])]
    target_columns = [x for x in range(to_model.shape[1] - lag_length*features, to_model.shape[1], features)]
    print(column_numbers)
    print("Target columns: ", target_columns)
    del column_numbers[-features*lag_length:]
    print(column_numbers)
    X = to_model.iloc[:, column_numbers]
    print(X)
    X.to_csv('temp.dat', sep=',')
    if np.any(np.isnan(X)):
        print("IS NAN HERE")

    #y1 = to_model.iloc[:, -features*3]
    y1 = to_model.iloc[:, target_columns].sum(axis=1)
    print("Target values", y1)
    if np.any(np.isnan(y1)):
        print("IS NAN HERE Y1")
    y2 = to_model.iloc[:, target_columns].sum(axis=1)
    print(y2)
    y3 = to_model.iloc[:, target_columns].sum(axis=1)
    print(y3)
    y_y1 = to_model.iloc[:, target_columns].sum(axis=1)
    y_v1 = to_model.iloc[:, target_columns].sum(axis=1)

    #for i in range(0, len(y1)):
    #    if y1[i] > 1:
    #        y1[i] = 1
    #    elif y1[i] < -1:
    #        y1[i] = -1
    #print(y1, y2, y3)
    return X, y1, y2, y3, y_y1, y_v1


def cut_tr_set_lstm(to_model, features=3):
    print("training_set", to_model)
    length=int(to_model.shape[1]/features-3)
    print("yay")
    column_numbers = [x for x in range(to_model.shape[1])]
    print(column_numbers)
    for i in range(features):
        del column_numbers[length*(i+1):length*(i+1)+3]
    #del column_numbers[length*2:length*2+3]
    #del column_numbers[length * 3:length * 3 + 3]
    print(column_numbers)
    X = to_model.iloc[:, column_numbers]
    print(X)
    y1 = to_model.iloc[:, length]
    print(y1)
    y2 = to_model.iloc[:, length+1]
    print(y2)
    y3 = to_model.iloc[:, length+2]
    print(y3)
    y_y1 = to_model.iloc[:, length+1]
    y_v1 = to_model.iloc[:, length+1]
    #print(y1, y2, y3)
    return X, y1, y2, y3, y_y1, y_v1


def simple_prep(r_vector, length, columns):
    all_columns = ['X', 'TIME', 'DELTAX', 'LEN', 'VOL', 'RATIO', 'VOLX', 'SIN', 'EMA1', 'EMA2','EMADIFF', 'TOEMA1', 'TOEMA2', 'DIFF1', 'DIFF2', 'DDIFF1', 'DDIFF2', 'DELTADIFF', 'DELTADDIFF', 'SUM']


    li = np.zeros(0)
    summa = 0
    for j in range(1, length+1):
        temp_values = {}
        x = r_vector[j, 0]
        time = r_vector[j, 1]*math.copysign(1, x)/2
        vol = r_vector[j, 2]
        vlen = math.sqrt(x ** 2 + time ** 2)*math.copysign(1, x)


        x_prev = r_vector[j - 1, 0]# if j > 0 else 1
        if x == 0  or x_prev == 0:
            x+=0.2
            x_prev+=0.2
            print("RATIO ", x, x_prev)
        deltax = x + x_prev
        summa = summa + x
        #print(summa)
        ratio = (x / abs(x_prev))/10
        ema1 = r_vector[j, 3]
        ema2 = r_vector[j, 4]
        emadiff = (ema1 - ema2)/10
        toema1 = x - ema1
        toema2 = x - ema2
        volx = vol / abs(x)
        sin = r_vector[j, 5]
        diff1 = r_vector[j, 6]
        diff2 = r_vector[j, 7]
        deltadiff = diff1 - diff2
        second_diff1 = r_vector[j, 8]
        second_diff2 = r_vector[j, 9]
        deltaddiff = second_diff1 - second_diff2
        temp_values['X'] = x
        temp_values['DELTAX'] = deltax
        temp_values['RATIO'] = ratio
        temp_values['TIME'] = time
        temp_values['VOLX'] = volx
        temp_values['LEN'] = vlen
        temp_values['VOL'] = vol
        temp_values['EMA1'] = ema1
        temp_values['EMA2'] = ema2
        temp_values['EMADIFF'] = emadiff
        temp_values['TOEMA1'] = toema1
        temp_values['TOEMA2'] = toema2
        temp_values['SIN'] = sin
        temp_values['DIFF1'] = diff1*10
        temp_values['DIFF2'] = diff2*10
        temp_values['DDIFF1'] = second_diff1
        temp_values['DDIFF2'] = second_diff2
        temp_values['DELTADIFF'] = deltadiff*100
        temp_values['DELTADDIFF'] = deltaddiff*100
        temp_values['SUM'] = summa/2


        #'RATIO', 'VOLX', 'SIN', 'DIFF1', 'DIFF2', 'DDIFF1', 'DDIFF2'
        tocheck = []



        #cub = (vol*x*time) ** (1. / 3.)
        list_to_append = []
        for col in all_columns:
            if col in columns:
                list_to_append.append(temp_values[col])
                tocheck.append(col)


        #print('list_to_append', list_to_append)

        if np.any(np.isnan(list_to_append)):
            print('ISNAN:', list_to_append)
            print(tocheck)
        #print('check', [x, emadiff, toema1, toema2, ratio, volx, sin])
        #li = np.append(li, [x, emadiff, toema1, toema2, ratio, volx, sin])#, ema1, ema2, diff1, diff2, second_diff1, second_diff2])
        li = np.append(li, list_to_append)#, ema1, ema2, diff1, diff2, second_diff1, second_diff2])

        #print([x, time, ema1, ema2, emadiff, ratio, volx, sin])
        #li = np.append(li, [x, ema1, ratio])


    return li

def get_mode(mode):
    match mode:
        case 'std':
            features = ['X', 'LEN', 'EMA1', 'EMA2', 'EMADIFF',  'Y', 'VOL', 'RATIO']
        case 'simple':
            features = ['X', 'EMA1', 'EMA2', 'EMADIFF', 'RATIO', 'VOLX']

    return features