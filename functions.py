#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import keras
from mnist import MNIST
import os.path
from cnn import *

def extract_data(path='data'):
  # Extrair Dados 
  mnist_data = MNIST(os.path.abspath(path))

  # Remodelar dados de treinamento
  X, Y = mnist_data.load_training()
  X_train = np.reshape(np.asarray(X, dtype=np.uint8), (60000, 28, 28, 1))
  Y_train = np.reshape(np.asarray(Y, dtype=np.uint8), (60000,))
  # Normalize os dados para obter melhores resultados
  X_train = X_train.astype('float32')/255
  # Converter rótulos em categóricos (0 -> [1 0 0 0 0 0 0 0 0 0])
  Y_train = keras.utils.to_categorical(Y_train, num_classes=10)

  # Remodelar dados de teste
  X, Y = mnist_data.load_testing()
  X_test = np.reshape(np.asarray(X, dtype=np.uint8), (10000, 28, 28, 1))
  Y_test = np.reshape(np.asarray(Y, dtype=np.uint8), (10000,))
  # Normalize os dados para obter melhores resultados
  X_test = X_test.astype('float32')/255
  # Converter rótulos em categóricos (5 -> [0 0 0 0 0 1 0 0 0 0])
  Y_test = keras.utils.to_categorical(Y_test, num_classes=10)
  return (X_train, Y_train), (X_test, Y_test)

def get_test_models():
  # Crie uma lista de parâmetros para testar
  models_params = [
    {
      'name': 'padrao-4-6-500',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 5},
        'second_layer': {'filters': 6, 'kernel_size': 5},
        'mlp_neurons': 500
      }
    },
    {
      'name': 'mudar_filtros-6-4-500',
      'parameters': {
        'first_layer': {'filters': 6, 'kernel_size': 5},
        'second_layer': {'filters': 4, 'kernel_size': 5},
        'mlp_neurons': 500
      }
    },
    {
      'name': 'mudar_mlp-4-6-700',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 5},
        'second_layer': {'filters': 6, 'kernel_size': 5},
        'mlp_neurons': 700
      }
    },
    {
      'name': 'mudar_mlp-4-6-300',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 5},
        'second_layer': {'filters': 6, 'kernel_size': 5},
        'mlp_neurons': 300
      }
    },
    {
      'name': 'mudar_kernel7-4-6-500',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 7},
        'second_layer': {'filters': 6, 'kernel_size': 7},
        'mlp_neurons': 500
      }
    },
    {
      'name': 'mudar_kernel3-4-6-500',
      'parameters': {
        'first_layer': {'filters': 4, 'kernel_size': 3},
        'second_layer': {'filters': 6, 'kernel_size': 3},
        'mlp_neurons': 500
      }
    },
  ]
  return models_params

def remove_marks(ax):
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

def train_models(models_params, X_train, Y_train, X_test, Y_test):
  data = {}
  for model_params in models_params:
    name = model_params['name']
    params = model_params['parameters']

    # Crie o modelo para o mnist
    model = MnistModel(name)
    model.generate_model(**params)
    # Treine o modelo e salve o modelo
    model.train(X_train, Y_train, epochs=20)
    model.save_model()
    print("Trained model {}.".format(name))

    # Carregar métricas
    history = model.history.copy()
    metrics = model.calculate_metrics(X_test, Y_test)
    metrics_names = model.get_metrics_names()

    # Estenda as métricas para todas as épocas
    for metric, values in history.items():
      for i in range(len(values), 20):
        values.append(values[len(values)-1])

    # Salve as métricas
    data[name] = (history, dict(zip(metrics_names, metrics)))
  return data

def load_models(models_params, X_test, Y_test):
  data = {}
  for model_params in models_params:
    name = model_params['name']
    params = model_params['parameters']

    # Carregue o modelo para o mnist
    model = MnistModel(name)
    model.load_model()
    print("Loaded model {}.".format(name))

    # Carregar métricas
    history = model.history.copy()
    metrics = model.calculate_metrics(X_test, Y_test)
    metrics_names = model.get_metrics_names()

    # Estenda as métricas para todas as épocas
    for metric, values in history.items():
      for i in range(len(values), 20):
        values.append(values[len(values)-1])

    # Salve as métricas
    data[name] = (history, dict(zip(metrics_names, metrics)))
  return data

def present_metrics(data):
  # Crie as figuras e eixos para traçar os dados
  loss = plt.figure("Loss").gca()
  accuracy = plt.figure("Accuracy").gca()
  precision = plt.figure("Precision").gca()

  for name, (history, metrics) in data.items():
    # Métricas de plotagem
    loss.plot(history['val_loss'], label=name)
    accuracy.plot(history['val_acc'], label=name)
    precision.plot(history['val_precision'], label=name)

    # Imprimir resultados de testes
    print(name)
    for name, metric in metrics.items():
      print("{}: {}".format(name, metric))
    print()

  # Embeleze os gráficos
  loss.set_xlabel("Epochs")
  loss.set_ylabel("Loss (Categorical Crossentropy)")
  loss.set_title("Loss Evolution")
  loss.set_ylim(0, 0.15)
  loss.set_xticks(list(range(20)))
  loss.legend()

  accuracy.set_xlabel("Epochs")
  accuracy.set_ylabel("Accuracy")
  accuracy.set_title("Accuracy Evolution")
  accuracy.set_ylim(0.95, 1)
  accuracy.set_xticks(list(range(20)))
  accuracy.legend()

  precision.set_xlabel("Epochs")
  precision.set_ylabel("Precision")
  precision.set_title("Precision Evolution")
  precision.set_ylim(0.95, 1)
  precision.set_xticks(list(range(20)))
  precision.legend()
  return loss, accuracy, precision

def save_all_figs():
  for label in plt.get_figlabels():
    fig = plt.figure(label)
    fig.savefig('img/'+label+'.png')

def visualize_filters(model):
  # Carregar filtros para exibição
  layers = model.get_filters()

  # Mostrar todos os filtros de neurônios da primeira camada em um único gráfico
  layer = 'first_conv'
  fig = plt.figure(layer+'-filters-all_neurons')
  axes = fig.subplots(2,2)
  axes = list(axes.flatten())
  filt = layers[layer][0]
  ax_num = 0
  for i in range(filt.shape[2]):
    for o in range(filt.shape[3]):
      axes[ax_num].matshow(filt[:,:,i:i+1,o:o+1].squeeze(), cmap='gray')
      axes[ax_num].set_title('{}-{}-{}'.format(layer, i, o))
      remove_marks(axes[ax_num])
      ax_num += 1

  # Mostre todos os 4 canais de cada neurônio da segunda camada em um único gráfico
  layer = 'second_conv'
  filt = layers[layer][0]
  for o in range(filt.shape[3]):
    fig = plt.figure(layer+'-filters-neuron_'+str(o))
    axes = fig.subplots(2,2)
    axes = list(axes.flatten())
    ax_num = 0
    for i in range(filt.shape[2]):
      axes[ax_num].matshow(filt[:,:,i:i+1,o:o+1].squeeze(), cmap='gray')
      axes[ax_num].set_title('{}-{}-{}'.format(layer, i, o))
      remove_marks(axes[ax_num])
      ax_num += 1

def visualize_activations(model, X):
  layers = {
    'first_conv': ((2,2), {}),
    'second_conv': ((2,3), {'fontsize':10.5}),
  }
  
  # Apresente as ativações como exemplo
  for layer, (subplots, title_properties) in layers.items():
    # Obtenha a ativação para a entrada especificada
    activation = model.get_activation(X, layer_name=layer)
    fig = plt.figure(layer+'-neurons_activation')
    axes = fig.subplots(*subplots)
    axes = list(axes.flatten())
    for neuron in range(activation.shape[3]):
      axes[neuron].matshow(activation[0:1,:,:,neuron:neuron+1].squeeze(), cmap='gray')
      axes[neuron].set_title('{}-neuron_{}'.format(layer, neuron), **title_properties)
      remove_marks(axes[neuron])

def visualize_wrong(model, X_test, Y_test, n_samples=5):
  prediction = np.round(model.predict(X_test))
  cond = (prediction != Y_test).any(axis=1)
  X_wrong = X_test[cond]
  Y_wrong = Y_test[cond]
  pred_wrong = prediction[cond]

  for i in range(n_samples):
    X = X_wrong[i:i+1,:,:,:].squeeze()
    Y = Y_wrong[i:i+1,:].squeeze().argmax()
    pred = pred_wrong[i:i+1,:].squeeze().argmax()

    fig = plt.figure('wrong_prediction_{}'.format(i))
    fig.gca().matshow(X, cmap='gray')
    remove_marks(fig.gca())

    fig.text(0.68,0.75+0.11-0.05,'Esperado:',fontsize=28)
    fig.text(0.68,0.5+0.11-0.05, str(Y),fontsize=100)
    fig.text(0.68,0.25+0.11,'Predito:',fontsize=28)
    fig.text(0.68,0.11, str(pred),fontsize=100)
    fig.subplots_adjust(left=0.04, right=0.68)


# In[ ]:




