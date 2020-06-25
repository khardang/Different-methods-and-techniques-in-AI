import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras





data = pickle.load(open("keras-data.pickle", "rb"),encoding='latin1')

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]
vocab_size = data["vocab_size"]
max_length = data["max_length"]
int_max = int(max_length/10)


#LSTM
x_train=keras.preprocessing.sequence.pad_sequences(x_train, maxlen=int_max, dtype='int32', padding='pre', truncating='pre', value=0.0)
x_test=keras.preprocessing.sequence.pad_sequences(x_test, maxlen=int_max, dtype='int32', padding='pre', truncating='pre', value=0.0)

y_train= np.asarray(y_train)
y_test = np.asarray(y_test)

#RNN model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size,embeddings_constraint=None, input_length=int_max, output_dim=64))

#Add a LSTM layer with one output units:
model.add(layers.LSTM(units=1, activation='tanh', recurrent_activation='sigmoid'))
#add a Dense layer with one output units:
model.add(layers.Dense(units=1, activation=None))

model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])

#Training on training data
training_model = model.fit(x_train,y_train,epochs=3)

#Evaluation with using the model on test data
evaluation = model.evaluate(x_test,y_test)
