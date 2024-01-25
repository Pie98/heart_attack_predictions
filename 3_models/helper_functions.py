import tensorflow as tf
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import datetime

class CSVLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename, experiment_name, overwrite=False):
        self.filename = filename
        self.experiment_name = experiment_name
        self.fieldnames = ['experiment', 'datetime', 'epoch', 'loss', 'accuracy', 'val_loss',
                           'val_accuracy']  # Aggiungi altre metriche secondo necessità
        self.first_time = overwrite

        if self.first_time:
            write_mode = 'w'
        else:
            write_mode = 'a'
        with open(self.filename, mode=write_mode, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if self.first_time:
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = {
            'experiment': self.experiment_name,
            'datetime': current_time,
            'epoch': epoch,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy'),
            # Aggiungi altre metriche secondo necessità
        }

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)


class CSVLoggerCallback_mae(tf.keras.callbacks.Callback):
    def __init__(self, filename, experiment_name, overwrite=False):
        self.filename = filename
        self.experiment_name = experiment_name
        self.fieldnames = ['experiment', 'datetime', 'epoch', 'loss', 'mae', 'val_loss',
                           'val_mae']  # Aggiungi altre metriche secondo necessità
        self.first_time = overwrite

        if self.first_time:
            write_mode = 'w'
        else:
            write_mode = 'a'
        with open(self.filename, mode=write_mode, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if self.first_time:
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = {
            'experiment': self.experiment_name,
            'datetime': current_time,
            'epoch': epoch,
            'loss': logs.get('loss'),
            'mae': logs.get('mae'),
            'val_loss': logs.get('val_loss'),
            'val_mae': logs.get('val_mae'),
            # Aggiungi altre metriche secondo necessità
        }

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)


#plotto la loss e accuracy curves separatamente
def plot_loss_curve(history):
  '''
  restituisce curve distinte per loss e accuracy
  '''
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  epochs = range(len(history.history['loss']))

  #plot loss
  plt.figure()
  plt.plot(epochs, loss ,label=['training_loss'])
  plt.plot(epochs, val_loss ,label=['val_loss'])
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()

  #plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy ,label=['training_accuracy'])
  plt.plot(epochs, val_accuracy ,label=['val_accuracy'])
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()

  return 0


import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), threshold=None, text_size=15):
    # Calcola la matrice di confusione e la normalizza
    mc = confusion_matrix(y_true, y_pred)
    mc_norm = mc.astype('float') / mc.sum(axis=1)[:, np.newaxis]

    n_classes = mc.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.matshow(mc, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes is None:
        classes = np.arange(n_classes)

    ax.set(
        title='Confusion Matrix',
        xlabel='Predicted Labels',
        ylabel='True Label',
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes)
    )
    # Ruota le etichette degli assi x e y di 45 gradi
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes, rotation=45)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    if threshold is None:
        threshold = (mc.max() + mc.min()) / 2

    for i, j in itertools.product(range(mc.shape[0]), range(mc.shape[1])):
        plt.text(j, i, f"{mc[i, j]} \n ({mc_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if mc[i, j] > threshold else "black",
                 size=text_size)

    return mc_norm

import os
def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img


# Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

    Parameters
    ---------
    timesteps : array of timesteps
    values : array of values across time
    format : style of plot, default "."
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (setting a value will index from end of timesteps & values)
    label : label to show on plot of values
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)