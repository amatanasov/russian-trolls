import numpy as np
import pandas as pd
import nltk
import tqdm

nltk.download('punkt')
nltk.download('wordnet')

from keras.layers import concatenate, Dropout, GlobalMaxPool1D, SpatialDropout1D, multiply
from keras.layers import Activation, Lambda, Permute, Reshape, RepeatVector, TimeDistributed
from keras.layers import Bidirectional, CuDNNGRU, Dense, Embedding, Input

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Bidirectional, Dropout, CuDNNGRU, Input
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

import numpy as np
import pandas as pd
import nltk
import tqdm

nltk.download('punkt')
nltk.download('wordnet')

from keras.layers import Dense, Embedding, Input, \
    Bidirectional, Dropout, CuDNNGRU, GRU, \
    Input, Activation, LSTM, GlobalMaxPool1D, BatchNormalization, GlobalMaxPooling1D, \
    Convolution1D, Conv1D, InputSpec, Flatten, GlobalAveragePooling1D, Concatenate, TimeDistributed, Lambda
from keras.layers.merge import add, concatenate

from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.model_selection import KFold

import os
from sklearn.model_selection import KFold
import tensorflow as tf



from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import concatenate, Dropout, GlobalMaxPool1D, SpatialDropout1D, multiply
from keras.layers import Activation, Lambda, Permute, Reshape, RepeatVector, TimeDistributed
from keras.layers import Bidirectional, CuDNNGRU, Dense, Embedding, Input

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Bidirectional, Dropout, CuDNNGRU, Input
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model


import numpy as np
import pandas as pd
import re
import random
import pandas as pd
import requests
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

import sys

import pandas as pd
import numpy as np
import os
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool
from keras.preprocessing import text, sequence
import os
import numpy as np
import pandas as pd
import re
import random
import pandas as pd
import requests
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import glob

PATH = "/social_bias_data/russian-troll-tweets/"

filenames = glob.glob(os.path.join(PATH, "*.csv"))

for file in filenames:
    print("Preprocessing {}".format(file))
df = pd.concat((pd.read_csv(f) for f in filenames))

left_troll = df[ df["account_category"] == "LeftTroll"]

right_troll = df[ df["account_category"] == "RightTroll"]

trolls = pd.concat([left_troll,right_troll])

trolls = trolls[["content","account_category"]]

trolls["account_category"] = trolls["account_category"].map({"LeftTroll":1,"RightTroll":0})

CLASSES = ["account_category"]


BATCH_SIZE = 256
DENSE_SIZE = 32
RECURRENT_SIZE = 64
DROPOUT_RATE = 0.3
MAX_SENTENCE_LENGTH = 500
OUTPUT_CLASSES = 1
MAX_EPOCHS = 18

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

train = trolls
del trolls

train["content"].fillna(NAN_WORD, inplace=True)

list_sentences_train = train["content"]

list_sentences_train = list_sentences_train.values

y_train = train[CLASSES].values

def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict

def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []
    with open(file_path) as f:
        for row in tqdm.tqdm(f.read().split("\n")[1:-1]):
            data = row.split(" ")
            word = data[0]
            embedding = np.array([float(num) for num in data[1:-1]])
            embedding_list.append(embedding)
            embedding_word_dict[word] = len(embedding_word_dict)

    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict

def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train

from keras.layers import GlobalAveragePooling1D, Reshape, Dense, multiply, add


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = -1
    filters = init._keras_shape[channel_axis]
    shape = (1, filters)

    se = GlobalAveragePooling1D()(init)
    se = Reshape(shape)(se)
    se = Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(se)

    output = multiply([init, se])

    return output

print("Tokenizing train data")

print("Loading embeddings...")
embedding_list, embedding_word_dict = read_embedding_list("/emb_data/glove.twitter.27B.100d.txt")
embedding_size = len(embedding_list[0])


tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

print("Preparing data...")

embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append([0.] * embedding_size)
embedding_word_dict[END_WORD] = len(embedding_word_dict)
embedding_list.append([-1.] * embedding_size)

embedding_matrix = np.array(embedding_list)

id_to_word = dict((id, word) for word, id in words_dict.items())

train_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_train,
    id_to_word,
    embedding_word_dict,
    MAX_SENTENCE_LENGTH)

X_train = np.array(train_list_of_token_ids)

kf = KFold(5, shuffle=True, random_state=2018)

fold = 0
test_predicts_list = []

for train, test in kf.split(X_train, y_train):

    lstm_input = Input(shape=(500,), name='lstm_input')

    x = Embedding(embedding_matrix.shape[0], (embedding_matrix.shape[1]),
                  weights=[embedding_matrix], trainable=False)(lstm_input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNGRU(units=128,
                               return_sequences=True))(x)

    y = Conv1D(64, 8, padding="same", kernel_initializer="he_uniform")(x)
    y = BatchNormalization()(y)
    y = Activation("elu")(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 5, padding="same", kernel_initializer="he_uniform")(y)
    y = BatchNormalization()(y)
    y = Activation("elu")(y)
    y = squeeze_excite_block(y)

    y = Conv1D(64, 3, padding="same", kernel_initializer="he_uniform")(y)
    y = BatchNormalization()(y)
    y = Activation("elu")(y)

    y = GlobalAveragePooling1D()(y)

    output_layer = Dense(1, activation="sigmoid")(y)

    model = Model(inputs=lstm_input, outputs=output_layer)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # Callbacks
    checkpointer = ModelCheckpoint(filepath="/output/weights.fold." + str(fold) + ".hdf5",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   monitor='val_loss',
                                   verbose=1)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    # train the model
    history = model.fit(X_train[train],
                        y_train[train],
                        epochs=20,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_train[test], y_train[test]),
                        callbacks=[earlystopper, checkpointer],
                        shuffle=True)

    #model.load_weights(filepath="/output/weights.fold." + str(fold) + ".hdf5", by_name=False)

    #predictions = model.predict(X_test, batch_size=BATCH_SIZE)
    #test_predicts_list.append(predictions)

    predictions = model.predict(X_train[test], batch_size=BATCH_SIZE)

    fold += 1

    K.clear_session()