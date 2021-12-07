"""
Team:
<<<<< TEAM NAME >>>>>
Authors:
<<<<< NOM COMPLET #1 - MATRICULE #1 >>>>>
<<<<< NOM COMPLET #2 - MATRICULE #2 >>>>>
"""

from wine_testers import WineTester
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import math
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras. initializers import TruncatedNormal 
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

class MyWineTester(WineTester):
    def __init__(self):
        model = Sequential()
        model.add(Dense(units = 128, input_shape=(9,))) 
        model.add(Activation('relu'))
        model.add(Dense(units = 128))
        model.add(Activation('relu'))
        model.add(Dense(units = 128))
        model.add(Activation('relu'))
        model.add(Dense(units = 128))
        model.add(Activation('relu'))
        model.add(Dense(units = 128))
        model.add(Activation('relu'))
        model.add(Dense(units = 128))
        model.add(Activation('relu'))                        
        model.add(Dense(units = 128))
        model.add(Activation('relu'))
        model.add(Dense(units = 10))
        model.add(Activation('softmax'))
                
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.model = model

    def train(self, X_train, y_train):
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        # train_features, val_features = train_test_split(X_train,  test_size=0.2, shuffle = False)
        # train_target, val_target = train_test_split(y_train,  test_size=0.2, shuffle = False)

        # print( 'train_target', len(train_target), 'train_features',  len(train_features) )
        # print( 'val_target  ',   len(val_target), 'val_features  ',    len(val_features) )


        df = pd.DataFrame(X_train)
        # df[1] = (df[1] == 'white').astype(int)
        
        # print(df)
        
        dfy = pd.DataFrame(y_train)
        
        # df = df.astype(float)
        # dfy = dfy.astype(float)
        # dfy = dfy.drop(columns=[0])
        # npx = df.to_numpy()
        # npy = dfy.to_numpy()
        # print(npy)
        
        # self.model.fit(npx, npy, epochs=30, verbose = 2, validation_split=0.2)


        df['is_white_wine'] = [1 if color == 'white' else 0 for color in df[1]]
        df = df.drop(columns=[0])
        df = df.drop(columns=[1])
        dfy = dfy.drop(columns=[0])



        df_data = pd.read_csv('./data/train.csv', sep=';')
        df_data = pd.get_dummies(df_data)
        dfx = df_data[df_data.columns.difference(['id', 'quality', 'color_red', 'color_white', 'fixed acidity', 'citric acid'], sort=False)]

    #     dfx = df_data[['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
        # dfx = df_data[[ 'volatile acidity', 'free sulfur dioxide',
        #             'sulphates', 'residual sugar', 'pH', 'total sulfur dioxide', 
        #             'chlorides', 'density', 'alcohol']]
        
        dfy = df_data['quality']


        x = np.array(dfx)
        y = np.array(dfy)       
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        y_true = y_val
        y_train, y_val = to_categorical(y_train, num_classes=10), to_categorical(y_val, num_classes=10)
        #fit
        h = self.model.fit(X_train_scaled, y_train, batch_size=64, epochs=300,
                          verbose=2, validation_data=(X_val_scaled, y_val))
                
        score = self.model.evaluate(X_val_scaled, y_val, verbose=0)

        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        
        samples = self.model.predict(X_val_scaled)
    
        test = np.argmax(samples, axis=1)

        predictions = []
        for idx, pred in enumerate(test):
            predictions.append([pred])


        good_pred = 0
        for i in range(len(predictions)):
            if predictions[i] == y_true[i]:
                good_pred += 1
            # else:
            #     print(str(predictions[i]) + ' : ' + str(y_true[i]))
        print('Accuracy: ', good_pred, ' / ', len(predictions), ' = ', good_pred/len(predictions))

        print(score)
        print(score)

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        df = pd.DataFrame(X_data)
        columnNames= ['id','color','fixed acidity','volatile acidity',
        'citric acid','residual sugar','chlorides','free sulfur dioxide',
        'total sulfur dioxide','density','pH','sulphates','alcohol','quality']
        
        for idz, c in enumerate(columnNames):
            print(str(idz) + ' : ' + c)
            df.rename(columns={idz: c})
        
        dfx = df[df.columns.difference([0, 1, 2, 4, 13, 'id', 'color', 'quality', 'color_red', 'color_white', 'fixed acidity', 'citric acid'], sort=False)]
        X_test_scaled = self.scaler.transform(dfx)
        samples = self.model.predict(X_test_scaled)
        print(dfx)
        test = np.argmax(samples, axis=1)
        predictions = []
        for idx, pred in enumerate(test):
            predictions.append([int(X_data[idx][0]), pred])
        return predictions
