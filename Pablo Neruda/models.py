from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.layers import Masking
import re

def words_to_punctuation(decoded_word):
    return decoded_word.replace('char_comma', '\b,')\
          .replace('char_exclamation_open', '\b¡')\
          .replace('char_exclamation_close', '\b!')\
          .replace('char_dot', '\b.')\
          .replace('char_dos_puntos', '\b:')\
          .replace('char_new_line', '\n')

def find_uncoded_stuff_and_retun_words_as_array(raw_text_cleanned):
    all_text = ''.join(raw_text_cleanned).lower()
    separators = re.compile('(?u)\\b\\w+\\b').split(all_text)
    # Imprime si encuentra cosas sin codificar
    for s in separators:
        for c in s:
            if c!=' ':
                print('char=',c, 'code=',ord(c))
    return re.findall('(?u)\\b\\w+\\b', all_text)
    
def replace_chars(line):
    return line.replace(' \n', '\n') \
        .replace('    ', ' ') \
        .replace('   ', ' ') \
        .replace('  ', ' ') \
        .replace('(', ' char_open_parent ') \
        .replace(')', ' char_close_parent ') \
        .replace('\n', ' char_new_line ') \
        .replace('.', ' char_dot ') \
        .replace(',', ' char_comma ') \
        .replace('¡', ' char_exclamation_open ') \
        .replace('!', ' char_exclamation_close ') \
        .replace('¿', ' char_question_open ') \
        .replace('?', ' char_question_close ') \
        .replace(':', ' char_dos_puntos ') \
        .replace('«', '"') \
        .replace('»', '"') \
        .replace("'", '"') \
        .replace('“', '"') \
        .replace('”', '"') \
        .replace(';', ' char_punto_y_coma ') \
        .replace('-', ' char_guion_medio ') \
        .replace('"', ' char_comillas ')


      #.replace('-', 'char_guion_medio')
          

def clean_raw_text_and_verify(raw_text):
    cont_elem = len(raw_text)
    i = 0
    raw_text_no_cr = []
    versos_count = 0
    soneto_nro = 0
    estructura = []
    while (i<cont_elem):
        if raw_text[i][:7] == 'Soneto ':
            # Es el titulo
            i += 1
            soneto_nro += 1
            # Saco enters despues de Soneto XX
            while raw_text[i]=='\n':
                i += 1
            raw_text_no_cr.append(' START_SONETO ')

        if raw_text[i]=='\n':
            # Fin estrofa
            raw_text_no_cr.append(replace_chars('\n'))
            estructura.append(versos_count)
            versos_count=0
            if len(estructura)==4:
                # Fin soneto
                if estructura!=[4, 4, 3, 3]:
                    print('Soneto', soneto_nro)
                    print('estructura', estructura)
                estructura = []
                raw_text_no_cr.append(' END_SONETO ')
                # Saco todos los enters antes de Soneto XX
                while raw_text[i]=='\n':
                    i += 1
            else: i += 1
                #print(raw_text[i])
        elif raw_text[i][:7] != 'Soneto ':
            # Es un verso
            versos_count = versos_count + 1
            raw_text_no_cr.append(replace_chars(raw_text[i]))
            i += 1
    if (raw_text_no_cr[-1]!=' END_SONETO '):
        raw_text_no_cr.append(' END_SONETO ')
    return raw_text_no_cr

def get_model_1(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size+1, 64, mask_zero = True))
    model.add(LSTM(100))
    model.add(Dense(vocab_size, activation="softmax"))
    return model

def get_model_1_statefull(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size+1, 64, mask_zero = True, batch_input_shape=(1,1)))
    model.add(LSTM(100, stateful= True))
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

def get_model_4(vocab_size, word_dim = 16, lstm_units=64):
    model = Sequential()
    model.add(Embedding(vocab_size+1, word_dim, mask_zero = True))
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation="softmax"))
    return model

def get_model_4_statefull(vocab_size, word_dim = 16, lstm_units=64):
    model = Sequential()
    model.add(Embedding(vocab_size+1, word_dim, mask_zero = True, batch_input_shape=(1,1)))
    model.add(LSTM(lstm_units, stateful=True)) #,
    model.add(Dense(vocab_size, activation="softmax"))
    return model