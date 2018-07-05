import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, Input

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

def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # This is the number of iterations taking into acount the step_size and the window_size
    N = int((len(text)-window_size)/step_size)
    # Get inputs and outputs
    for k in range(N):
        i = k*step_size
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        
    return inputs,outputs

def encode_io_pairs(text,chars, window_size,step_size):
    num_chars = len(chars)
    chars_to_indices = dict((c, i) for i, c in enumerate(chars))
    # cut up text into character input/output pairs
    inputs, outputs = window_transform_text(text,window_size,step_size)
    
    # create empty vessels for one-hot encoded input/output
    X = np.zeros((len(inputs), window_size, num_chars), dtype=np.bool)
    y = np.zeros((len(inputs), num_chars), dtype=np.bool)
    
    # loop over inputs/outputs and tranform and store in X/y
    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            if char not in chars_to_indices:
                char = ' '
            X[i, t, chars_to_indices[char]] = 1
        out_char = outputs[i]
        if out_char not in chars_to_indices:
            out_char = ' '
        y[i, chars_to_indices[out_char]] = 1
    return X,y

def get_deep_rnn(input_shape, dense_units = 80, LSTM_units_1=200, LSTM_units_2=200, dropout_p=0.2, stateful=False, verbose=True):
    # Model definition
    if verbose:
        print("input shape = ",input_shape)
    model = Sequential()
    if (stateful):
        model.add(LSTM(LSTM_units_1, 
                       batch_input_shape=(1,input_shape[0],input_shape[1]), 
                       return_sequences=True, 
                       name='lstm_1',
                       dropout=dropout_p, 
                       recurrent_dropout=dropout_p, 
                       stateful=stateful))
    else:    
        model.add(LSTM(LSTM_units_1, input_shape=input_shape, return_sequences=True, name='lstm_1',
                       dropout=dropout_p, recurrent_dropout=dropout_p, stateful=stateful))
    model.add(LSTM(LSTM_units_2, dropout=dropout_p, recurrent_dropout=dropout_p, name='lstm_2', stateful=stateful))
    model.add(Dense(dense_units, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    if verbose:
        model.summary()
    return model

def sample(a, temperature=1.0, verbose = False, return_dist=False):
    a = np.array(a)
    a = a/a.sum()
    a = a**(1/temperature)
    p_sum = a.sum()
    sample_temp = a/p_sum 
    if verbose:
        print(sample_temp)
    #sample_temp = sample_temp*(sample_temp>1e-4)
    choices = range(len(a))
    if return_dist:
        return np.random.choice(choices, p=sample_temp), sample_temp
    else:
        return np.random.choice(choices, p=sample_temp)
    #return np.argmax(np.random.multinomial(1, sample_temp, 1))

def chars_to_one_hot(sentence, chars, chars_to_indices, window_size):
    num_chars = len(chars)
    size = max(len(sentence),window_size)
    X = np.zeros((1, size, num_chars), dtype=np.bool)
    for t, char in enumerate(sentence):
        if char not in chars_to_indices:
            char = ' '
        else:
            X[0, t + size - len(sentence), chars_to_indices[char]] = 1
    return X