#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import keras
import json
import keras_metrics as km
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout

class MnistModel(object):
  def __init__(self, name):
    # Save the model name for future use
    self.name = name

  def generate_model(
      self,
      first_layer = {'filters': 4, 'kernel_size': 5},
      second_layer = {'filters': 6, 'kernel_size': 5},
      mlp_neurons = 500,
      verbose=0):
    # Define a model
    model = Sequential()

    # Add a convolutional layer
    model.add(Conv2D(**first_layer, input_shape=(28,28,1), padding="same", data_format="channels_last", name='first_conv'))
    # a pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Add a convolutional layer
    model.add(Conv2D(**second_layer, padding="same", name='second_conv'))
    # a pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Make matrix become flat
    model.add(Flatten())

    # Add a Multilayer Perceptron for classification
    model.add(Dense(mlp_neurons))
    # Add the output layer
    model.add(Dense(10, activation="softmax"))
    # Add a softmax activation layer
    # model.add(Activation("softmax"))

    # Finish creating the model
    model.compile(
      loss=keras.losses.categorical_crossentropy,
      optimizer=keras.optimizers.Adadelta(),
      metrics=['accuracy', km.precision()]
    )

    if verbose >= 1:
      print(model.summary())
    self.model = model

  def train(
      self,
      X_train, Y_train,
      X_valid=None, Y_valid=None,
      validation_split=0.2, batch_size=50, epochs=20,
      verbose=1):
    if verbose >= 1:
      print("Training model {}.".format(self.name))
    es = EarlyStopping(
      monitor='val_acc',
      mode='max',
      min_delta=0.001,
      patience=5
    )
    cp = ModelCheckpoint(
      self.save_name(suffix='best'),
      monitor='val_acc',
      mode='max',
      save_best_only=True
    )
    params = {
      'x': X_train, 'y': Y_train,
      'batch_size': batch_size,
      'epochs': epochs,
      'verbose': 1,
      'callbacks': [es],
    }
    if X_valid is None or Y_valid is None:
      history = self.model.fit(
        **params,
        validation_split=validation_split
      )
    else:
      history = self.model.fit(
        **params,
        validation_data=(X_valid, Y_valid)
      )
    self.history = history.history
    return history

  def predict(self, X_test):
    return self.model.predict(X_test, batch_size=50)

  def calculate_metrics(self, X_test, Y_test):
    return self.model.evaluate(X_test, Y_test, verbose=0)

  def get_metrics_names(self):
    return self.model.metrics_names[:]

  def save_name(self, suffix=None):
    if suffix == None:
      name = 'models/{}.h5'.format(self.name)
    else:
      name = 'models/{}_{}.h5'.format(self.name, suffix)
    return name

  def save_model(self):
    self.model.save(self.save_name())
    history_name = self.save_name().split('.')[0] + '.json'
    with open(history_name, 'w') as history_file:
      json.dump(self.history,history_file)

  def load_model(self, verbose=0):
    self.model = load_model(
      self.save_name(),
      custom_objects={'binary_precision': km.precision()}
    )
    if verbose >= 1:
      print(self.model.summary())
    history_name = self.save_name().split('.')[0] + '.json'
    with open(history_name, 'r') as history_file:
      self.history = json.load(history_file)

  def get_activation(self, X, layer_name='first_conv'):
    assert layer_name == 'first_conv' or layer_name == 'second_conv'
    # Creates model for getting the convolutional layers activation
    view_layer = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
    # Calculates the convolutional layer output
    return view_layer.predict(X)

  def get_filters(self):
    filters = {
      'first_conv': self.model.get_layer('first_conv').get_weights(),
      'second_conv': self.model.get_layer('second_conv').get_weights()
    }
    return filters


  @staticmethod
  def convert_label(Y :np.ndarray):
    # Create matrix with number of examples lines and 10 columns (10 possible digits 0-9)
    new = np.zeros((*Y.shape, 10), dtype=np.uint8)
    # For each line select the label column and mark as 1
    new[np.arange(Y.shape[0]), Y] = 1
    return new


# In[ ]:




