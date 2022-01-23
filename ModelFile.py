from matplotlib import pyplot as plt
import numpy as np
import copy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix
import itertools


def scaling(trainX, testX, k):
    """Scale the samples"""
    # Keep the values that appear at least to the 1% of the images in the train set
    threshold = 0.01
    unique, counts = np.unique(trainX, return_counts=True)
    counts_min = threshold * len(trainX)
    freq_values = unique[np.where(counts >= counts_min)]
    scaler_fitting_set = copy.deepcopy(trainX)
    for i in range(len(trainX)):
        if np.max(trainX[i]) > freq_values[-1]:
            scaler_fitting_set[i] = np.zeros((2 ** k, 2 ** k, 3))

    scaler = MinMaxScaler()
    scaler = scaler.fit(scaler_fitting_set.reshape(-1, 1))
    trainX_scaled = scaler.transform(trainX.reshape(-1, 1))
    trainX_scaled = trainX_scaled.reshape(np.shape(trainX))
    testX_scaled = scaler.transform(testX.reshape(-1, 1))
    testX_scaled = testX_scaled.reshape(np.shape(testX))

    return trainX_scaled, testX_scaled


def train_test(images, label_sequence, k):
    # Train/Test split
    stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=30)
    for train_idx, test_idx in stratSplit.split(images, label_sequence):
        x_train_without_scaling = images[train_idx]
        y_train = label_sequence[train_idx]
        x_test_without_scaling = images[test_idx]
        y_test = label_sequence[test_idx]

        x_train_scaled, x_test_scaled = scaling(x_train_without_scaling, x_test_without_scaling, k)

        return x_train_scaled, y_train, x_test_scaled, y_test


def model(k):
  input_layer = Input(shape=(2**k,2**k,3))
  conv1 = Conv2D(64, 2, padding = 'same', activation = 'selu')(input_layer)
  drop1 = Dropout(0.2)(conv1)
  batch1 = BatchNormalization()(drop1)
  pool = MaxPooling2D(2)(batch1)
  conv2 = Conv2D(128, 2, padding = 'same', activation = 'selu')(pool)
  drop2 = Dropout(0.2)(conv2)
  batch2 = BatchNormalization()(drop2)
  flat = Flatten()(batch2)
  pred = Dense(1, activation = 'sigmoid')(flat)

  cnn = Model(input_layer, pred)

  return cnn


def evaluation(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()

    accuracy = np.round((TP + TN) / (TP + TN + FP + FN), 2)
    specificity = np.round(TN / (TN + FP), 2)
    precision = np.round(TP / (TP + FP), 2)
    recall = np.round(TP / (TP + FN), 2)
    F1 = np.round(2 * ((precision * recall) / (precision + recall)), 2)

    metrics = {"Accuracy": accuracy, "Specificity": specificity, "Precision": precision, "Recall": recall, "F1": F1}

    return metrics


def plot_confusion_matrix(y_true, y_pred, title, Y_lbl, cmap=plt.cm.Blues):
    classes = np.unique(Y_lbl)
    labels = range(len(classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm2 = np.round(cm / cm.sum(axis=1)[:, None] * 100, decimals=1)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm2, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
    plt.title(title + " confusion matrix", fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm2.max() / 2.
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        plt.text(j, i, cm2[i, j],
                 horizontalalignment="center",
                 color="white" if cm2[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)

    plt.show()