# -*- coding: utf-8 -*-

import pandas as pd
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from tensorflow import keras

data_train = pd.read_csv('csv_data/train-v3.csv')
X_train = data_train.drop(['price','id'],axis=1)
Y_train = data_train['price'].values

data_valid=pd.read_csv('csv_data/valid-v3.csv')
X_valid = data_valid.drop(['price','id'],axis=1)
Y_valid = data_valid['price'].values

data_test = pd.read_csv('csv_data/test-v3.csv')
X_test = data_test.drop(['id'],axis=1)

mean,std=X_train.mean(axis=0,),X_train.std(axis=0)
X_train=(X_train-mean)/std
X_valid=(X_valid-mean)/std
X_test=(X_test-mean)/std



model =Sequential()
model.add(Dense(128,input_dim=X_train.shape[1],kernel_initializer='random_normal',activation='relu'))
model.add(Dense(64,kernel_initializer='normal',activation='relu'))
model.add(Dense(32,kernel_initializer='normal',activation='relu'))
model.add(Dense(1))
adam=keras.optimizers.Adam(learning_rate=0.008,beta_1=0.9, beta_2= 0.99, epsilon= None, decay=0.0, amsgrad= False)
model.compile(loss='MAE',optimizer=adam)

history = model.fit(X_train,Y_train,validation_data=(X_valid,Y_valid),
                  
                  epochs=250,batch_size=128,verbose=1)

losses=pd.DataFrame(model.history.history)
losses.plot()
model.summary()


pred=model.predict(X_test)

with open('csv_data/house_pred_keras.csv','w')as f:
  f.write('id,price\n')
  for i in range(len(pred)):
    f.write(str(i+1)+','+str(float(pred[i]))+'\n')
