# Ref https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import pandas as pd
from gensim import corpora
import os
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json

from sklearn.metrics import mean_squared_error

train_csv = "./data/train.csv"
test_csv = "./data/test.csv"
dictionary_file = "./data/datadict.dict"
train_id_csv = "./data/train_id.csv"
test_id_csv = "./data/test_id.csv"
lstm_json_model = "./data/lstm.json"
lstm_wts_model = "./data/lstm.wts"
prediction_file_csv = "./data/predictions.csv"


def prep_id_dataframes():

    train_file_df = pd.read_csv(train_csv, encoding='cp1252',header=None)
    test_file_df = pd.read_csv(test_csv, encoding='cp1252', header=None)
    train_file_df.columns = ['token', 'pos', 'iob']
    test_file_df.columns = ['token', 'pos']

    train_x_words = train_file_df['token'].tolist()
    train_y_words = train_file_df['iob'].tolist()
    test_x_words = test_file_df['token'].tolist()

    dictionary = corpora.Dictionary([train_x_words + test_x_words])
    dictionary.compactify()
    dictionary.save(dictionary_file)
    dictionary = corpora.Dictionary.load(dictionary_file)

    train_x_ids = [dictionary.token2id[tok] for tok in train_x_words]
    test_x_ids = [dictionary.token2id[tok] for tok in test_x_words]

    # GIVE IOB tags, the ids more than range of dictionary ids
    max_id = len(dictionary.token2id.values()) + 100
    iob2id = {'B-LEGAL':max_id,'I-LEGAL':max_id+1,'O':max_id+1}
    train_y_ids = [iob2id[tok] for tok in train_y_words]

    # Prep ID dataframe
    train_id_df = pd.DataFrame({'x': train_x_ids, 'y': train_y_ids})
    test_id_df = pd.DataFrame({'x': test_x_ids})

    train_id_df.to_csv(train_id_csv,header=None,index=False)
    test_id_df.to_csv(test_id_csv,header=None,index=False)

    return train_id_df, test_id_df

def prep_lstm_model(train_x, train_y):

    # split into train and test sets of TRAINING SET ONLY
    train_size = int(len(train_x) * 0.67)
    test_size = len(test_x) - train_size

    train_split_x  = train_x[0:train_size]
    train_split_y  = train_y[0:train_size]
    test_split_x = train_x[train_size:len(train_x)]
    test_split_y = train_y[train_size:len(train_y)]

    trainX = np.array(train_split_x)
    trainY = np.array(train_split_y)
    testX = np.array(test_split_x)
    testY = np.array(test_split_y)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, 1)) # (samples or entries, timestep=1 at a time, num features = 1 being single int id)
    testX = np.reshape(testX, (testX.shape[0], 1, 1))

    # create and fit the LSTM network
    look_back = 1 # entries to look backwards, for that X needs to be reformatted x1,x2,...xn for each outcome
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, batch_size=1, verbose=2,nb_epoch=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    return model

def save_model(md,js,wt):
    # serialize model to JSON
    model_json = md.to_json()
    with open(js, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    md.save_weights(wt)
    print("Saved model to disk")

def open_model(js, wt):
    json_file = open(js, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    md = model_from_json(loaded_model_json)
    # load weights into new model
    md.load_weights(wt)
    print("Loaded model from disk")
    return md

if __name__ == "__main__":

    train_df = None
    test_df = None
    if os.path.isfile(train_id_csv):
        print("Loading Dataframes")
        train_df = pd.read_csv(train_id_csv, header=None)
        test_df = pd.read_csv(test_id_csv, header=None)
    else:
        train_df, test_df = prep_id_dataframes()

    train_df.columns = ['x','y']
    test_df.columns = ['x']
    # print(train_df.head())
    # print(test_df.head())
    train_x = train_df['x'].values
    train_y = train_df['y'].values
    test_x = test_df['x'].values
    # print(train_x)
    # print(train_y)
    # print(test_x)

    model = None
    if os.path.isfile(lstm_json_model):
        model = open_model(lstm_json_model, lstm_wts_model)
    else:
        model = prep_lstm_model(train_x,train_y)
        save_model(model,lstm_json_model, lstm_wts_model)

    test_x = np.reshape(test_x, (test_x.shape[0], 1, 1)) # (samples or entries, timestep=1 at a time, num features = 1 being single int id)

    final_prediction = model.predict(test_x)
    test_df['y'] = final_prediction
    test_df.to_csv(prediction_file_csv, index=None, header=None)




