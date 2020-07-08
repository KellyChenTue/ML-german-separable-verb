import csv,sys
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import keras.backend as K
# fix random seed for reproducibility
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
import collections
from collections import Counter


csv.field_size_limit(sys.maxsize)
def read_data(filename, external = None):

    X= []
    y = []
    temp_sentence = []
    temp_label = []
    particle_lst = ["auf","aus", "zu", "durch", "gegen", "um", "mit", "nach", "an", "hinter", "hin", "neben", "her"]
    ending_word = [".","!",","]

    with open(filename, 'r') as csv_readerfile:
        reader = csv.reader(csv_readerfile, delimiter='\t')
        next(reader,None)


        for r in reader:
            if not r :
                temp_sentence = []

            if r and not str(r[0]).startswith("#"):
                temp_sentence.append(r[1]) # save the word of a sentence
                #print(r)



                if len(r) > 6 and r[1] in particle_lst :
                    temp_label.append(r[7])

                    #
                    # if r[7] == "compound:prt" :
                    #     y.append("1")
                    # elif r[7] == "case":
                    #      y.append("2")


                if r[1] in ending_word: #

                    if len(temp_label) ==1:
                        if temp_label[0] == "compound:prt" or temp_label[0] == "case": #only save the sentence which has only one label
                            X.append(temp_sentence)
                            y.append(temp_label[0])
                    temp_sentence = []
                    temp_label = []
    #
    #
    # print("________________________________________________")
    # print(X)
    # print(len(X))
    # print(y)
    # print(len(y))
    # print(Counter(y))

    return X, y

def read_pure_sent(filename, classes, limit_len_sent):
    data_list = []
    label_list = []

    with open(filename, 'r') as f:
        reader = f.readlines()# test with 5 sentences
        for r in reader:
            sent = r.split('\t')[1]
            # if ends with a label in our classes, append it into data_list and label_list
            token = nltk.word_tokenize(sent)
            if token[-2] in classes and len(token) < limit_len_sent:
                label_list.append(token[-2])
                # change label into  "______", then append
                token[-2]= "______"
                token.pop(0) #pop the first word, because it appears in many sents, maybe drop the acc
                data_list.append(token)
    return data_list, label_list

def split_and_encode(X,y):


    ## split
    X_train = X[:int(len(X)*0.8)+1]
    X_test = X[int((len(X)*(0.8)+1)):]
    #print(len(X_train), len(X_test), len(X))
    y_train = y[:int(len(X)*0.8)+1]
    y_test = y[int((len(X)*(0.8)+1)):]
    print(len(y_train), len(y_test), len(y))
    print(y_train)
    print(y_test)

    # encode sentences # (num_words=vocab_size)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train + X_test)
    word_index = tokenizer.word_index

    vocab_size= len(word_index)+1
    dict(list(word_index.items())[0:10])
    train_sequences = tokenizer.texts_to_sequences(X_train)
    print(train_sequences[0])

    train_padded = sequence.pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
    print(len(train_sequences[0]))
    print(len(train_padded[0]))
    print(train_padded[0])

    print(len(train_sequences[1]))
    print(len(train_padded[1]))

    print(len(train_sequences[10]))
    print(len(train_padded[10]))

    validation_sequences = tokenizer.texts_to_sequences(X_test)
    validation_padded = sequence.pad_sequences(validation_sequences, maxlen=max_length, padding="post",
                                      truncating="post")

    print(len(validation_sequences))
    print(validation_padded.shape)
    print(validation_padded)

    # encode labels

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(y)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(y_train))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(y_test))
    print("tr_label")
    print(training_label_seq[0])
    print(training_label_seq[1])
    print(training_label_seq[10])
    print(training_label_seq.shape)
    #print(collections.Counter(y).keys())
    #print(len(collections.Counter(y).keys()))


    print(validation_label_seq[0])
    print(validation_label_seq[1])
    print(validation_label_seq[2])
    print(validation_label_seq.shape)


    #X_train = sequence.pad_sequences(X_train, maxlen = max_length)
    #X_test = sequence.pad_sequences(X_test, maxlen = max_length)



    return vocab_size,train_padded, training_label_seq, validation_padded, validation_label_seq


def train(train_padded, training_label_seq, validation_padded, validation_label_seq):
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(embedding_dim, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(embedding_dim, activation="relu"))
    model.add(Dense(count_classes+1, activation="softmax"))

    """ 
    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
        # input_dim : how many distinct vocabulary in the data?
        # output_dim : vector space of the vocabulary
        # input_length : how many words are there for one input. the longest.
        # output = 2D vector space. Before adding a Dense layer, must use Flatten() to turn it back to 1D
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        #tf.keras.layers.Flatten(),

        #tf.keras.layers.LSTM(embedding_dim, dropout=0),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, dropout=0.01)),
        #tf.keras.layers.GRU(16),

        # use ReLU in place of tanh function since they are very good alternatives of each other.
        #tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(embedding_dim, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #tf.keras.layers.Dense(embedding_dim, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)),

        # Add a Dense layer with "number of classes" units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(num_classes+1, activation='softmax')
    ])



    model.summary()

    #optimizer
    Adam(lr=0.001,  beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
    num_epochs = 500
    batch_size = 16
    #print(K.eval(model.optimizer.lr))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
        update_freq='epoch',  embeddings_freq=0,
        embeddings_metadata=None
    )
    model.fit(train_padded, training_label_seq, batch_size=batch_size ,epochs=num_epochs,callbacks=[callback],
                        validation_data=(validation_padded, validation_label_seq), verbose=2)

def count_classes(y):
    count = 0
    for item, count in collections.Counter(y).items():
        print(count)

    print("count classes: " +str(len(collections.Counter(y))))
    return count

if __name__ == '__main__':
   X, y = read_data("/Users/chenpinzhen/snlp2019/a3-a3-kelly/UD_German-GSD/de_gsd-ud-train.conllu")
   X2, y2 = read_data("/Users/chenpinzhen/snlp2019/a3-a3-kelly/UD_German-GSD/de_gsd-ud-test.conllu")
   X3, y3 = read_data("/Users/chenpinzhen/snlp2019/a3-a3-kelly/UD_German-GSD/de_gsd-ud-dev.conllu")
   X = X + X2 + X3
   y = y+ y2 + y3

    # when there is aw word in the list[auf, aus...], is it a compound or a preposition?
    # 0. compound:prt 1. case

   # data preprocess:
   # if the sentence contains compound:prt or case, store the whole sentence to X, store the label 0 or 1 for Y







   num_classes = 2


   # extract most common N classes
   # most_common = collections.Counter(y).most_common(num_classes)
   # most_com_par = []
   # for most in most_common:
   #     most_com_par.append(most[0])
   # print("the classes to identify are: ")
   # print(most_com_par)
   # temp_X = []
   # temp_y = []
   # for i, label in enumerate(y):
   #     if label in most_com_par and len(X[i])<limit_len_sent:
   #         temp_X.append(X[i])
   #         temp_y.append(y[i])
   # X = temp_X
   # y = temp_y

   """
   # read more data , based on the labels we already have

   X2, y2 = read_pure_sent("/Users/chenpinzhen/snlp2019/deu-de_web-public_2019_1M-sentences.txt", most_com_par, limit_len_sent)
   print("X2, y2")
   print(len(X2),X2)
   print(len(y2),y2)
   X =  X + X2
   y =  y + y2
   """

   print(len(X), X)
   print(len(y), y)


   max_length = len(max(X, key=len))
   #max_length = 20
   print("max length of a sent")
   print(max_length)

   vocab_size, train_padded, training_label_seq, validation_padded, validation_label_seq = split_and_encode(X,y)
   vocab_size = vocab_size  # how many distinct words?
   embedding_dim = round(vocab_size**0.25)  # 978 #size of the embedding vectors


   print("max value")
   print(max(training_label_seq))
   print(training_label_seq[3])
   print(type(training_label_seq))
   print("train_padded shape")
   print(train_padded.shape)
   print(validation_padded.shape)

   train(train_padded, training_label_seq, validation_padded, validation_label_seq)
   print(set(y))