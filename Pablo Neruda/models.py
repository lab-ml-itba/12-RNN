from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

def get_model_1(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 64))
    model.add(LSTM(100))
    model.add(Dense(vocab_size, activation="softmax"))
    return model

def get_model_2(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 10))
    model.add(LSTM(100))
    model.add(Dense(vocab_size, activation="softmax"))
    return model

def get_model_3(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 10))
    model.add(LSTM(50))
    model.add(Dense(vocab_size, activation="softmax"))
    return model