# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 13:26:01 2017
 
@author: Paul Stafford
"""
import numpy as np
import pandas
from keras.models import Sequential
#from keras.callbacks import TensorBoard
from keras.layers import Dense,Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
 
np.set_printoptions(threshold=np.nan)
np.random.seed(7)
 
# load dataset
dataframe = pandas.read_csv("GBPUSD LSTM shape.csv", header=None)
dataset = dataframe.values
num_features = dataset.shape[1]
train_test_percent = .75
look_back = 8   # AKA sequence length = window size
 
# scale features in entire data set 
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset) 
 
# This function divides a dataset into multiple overlappingsequences of length "look-back"
# and takes the first column value as the target (advanced one time)
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0]) # Y value is first col of next time
    return np.array(dataX), np.array(dataY)
 
def plot_results(predicted_data,true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data,label = 'True Data')
    plt.plot(predicted_data,label = 'Prediction')
    plt.legend()
    plt.show()
 
def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted
 
# split into train and test sets
train_size = int(len(dataset) * train_test_percent)   
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
 
trainX, trainY = create_dataset(train, look_back)  
testX, testY = create_dataset(test, look_back)
 
# reshape input to be  [total samples, values in each sequence, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], num_features))
testX = np.reshape(testX, (testX.shape[0],testX.shape[1], num_features))
 
# build model and fit data
layers = [20, 30, 1] # makes it easy to add  additional layers
 
model = Sequential()
model.add(LSTM(
        layers[0], 
        input_shape=(None,num_features),
        return_sequences= True))
model.add(Dropout(0.2))
         
model.add(LSTM(
        layers[1],
        return_sequences=False))
model.add(Dropout(0.2))
 
model.add(Dense(layers[2],activation ='sigmoid'))
 
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(trainX, trainY,nb_epoch=500,  verbose = 0)
print(model.summary())
# make predictions
trainPredict = model.predict(trainX)
trainPredict = np.reshape(trainPredict, (trainPredict.size,))
plot_results(trainPredict,trainY)
 
 
#testPredict = model.predict(testX)
#testPredict = np.reshape(testPredict, (testPredict.size,))
#plot_results(testPredict,testY)
 
#print(scaler.inverse_transform(dataset))
#print(dataset[look_back], trainPredict[0]*(scaler.data_max_[0]-scaler.data_min_[0])+scaler.data_min_[0])