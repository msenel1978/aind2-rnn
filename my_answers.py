import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from string import ascii_lowercase

# DONE: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[i:i+window_size] for i in range(len(series)-window_size)]
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    rnn_model = Sequential()
    rnn_model.add(LSTM(units = 5, input_shape = (window_size, 1)))
    rnn_model.add(Dense(units = 1))

    return rnn_model


### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    # Keep punctuation and lower case ASCII	
    keep_char = punctuation + [ch for ch in ascii_lowercase] 
    
    # Keep if char is in keep_char. Otherwise, replace with space
    return ''.join([c if c in keep_char else ' ' for c in text])

### DONE: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
