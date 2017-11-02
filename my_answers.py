import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Lenght of series
    N = series.shape[0]
    
    # Generate inputs and outputs
    for i in range(N-window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # Model definition
    model = Sequential()
    # Layer 1
    model.add(LSTM(5, input_shape = (window_size,1)))
    # Output layer
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
import re
def cleaned_text(text):
    # Just kept it from original code
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    # Used a regular expression to just permit the list of letters and symbols: abcdefghijklmnopqrstuvwxyz!,.:;?
    text = re.sub(r'[^abcdefghijklmnopqrstuvwxyz!,.:;?]', ' ', text)
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # This is the number of iterations taking into acount the step_size and the window_size
    N = int((len(text)-window_size)/step_size)+1
    
    # Get inputs and outputs
    for k in range(N):
        i = k*step_size
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # Model definition
    model = Sequential()
    # First layer
    model.add(LSTM(200, input_shape =  (window_size,num_chars)))
    # Second layer
    model.add(Dense(num_chars))
    # Output layer
    model.add(Activation('softmax'))
    return model
