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


class MyWineTester(WineTester):
    def __init__(self):
        model = Sequential()

        # initial values
        features = Input(shape = (12,)) # number of columns - outcome

        # Input layer + hidden layer 1
        input_layer = Dense(32, activation='relu', use_bias=True)(features)

        # Hidden layers > 1
        h2 = Dense(64, activation='relu', use_bias=True)(input_layer)
        x = layers.Dropout(0.5)(h2)
        h3 = Dense(512, activation='relu', use_bias=True)(x)
        h4 = Dense(32, activation='relu', use_bias=True)(h3)

        # Output layer
        out = Dense(10, activation='softmax', use_bias=True)(h4)
        
        model = Model(inputs = [features],
                    outputs= [out])

        #optimizer
        adam = Adam()
    
        model.compile(loss  = 'categorical_crossentropy',
                  optimizer = adam,
                  metrics=[
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                            ])
        # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


        self.scaler = StandardScaler()
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
        
        X_train, X_val, y_train, y_val = train_test_split(df, dfy, test_size=0.2, random_state=42)

        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        y_train, y_val = to_categorical(y_train, num_classes=10), to_categorical(y_val, num_classes=10)
        #fit
        history = self.model.fit(X_train_scaled, y_train, epochs=80, validation_data=(X_val_scaled, y_val), shuffle=False, verbose=2)
        print("hold")

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
        
        df['is_white_wine'] = [1 if color == 'white' else 0 for color in df[1]]
        df = df.drop(columns=[0])
        df = df.drop(columns=[1])
        X_test_scaled = self.scaler.transform(df)
        samples = self.model.predict(X_test_scaled)
        predicts = np.maximum(samples,0)        
        test       = np.argmax(predicts, axis=1)
        predictions = []
        for idx, pred in enumerate(test):
            predictions.append([int(X_data[idx][0]), pred])
        return predictions
