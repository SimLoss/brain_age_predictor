"""Module for Deep Dense Network implementation."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import  mean_absolute_error
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     StratifiedKFold)
import tensorflow as tf
from tensorflow.keras.layers import (Dense,
                                     Dropout,
                                     Input,
                                     BatchNormalization)
from tensorflow.keras.models import Model
from sklearn.base import BaseEstimator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############################
class AgeRegressor(BaseEstimator):
    """
    Class describing a Deep Dense Network used as a linear regressor.
    The class inherits from BaseEstimator for an easy implementation into
    scikit's pipeline and grid search.
    Linear regression is performed using 'mean absolute error' as loss func
    to minimize.

    Attributes
    ----------

    learning_rate : float
                    Learning rate value.

    batch_size : int
                Batch size value.

    dropout_rate: float
                Dropout rate value to be passed to dropout layer.

    epochs: int
            Number of iterations on entire dataset.

    verbose: bool, DEFAULT=False
            If True, prints the model's summary.

    model : object
            Compiled model.

    Example
    -------
         Layer (type)                Output Shape              Param #
    =================================================================
     input_57 (InputLayer)       [(None, 128)]             0

     dense_280 (Dense)           (None, 64)                8256

     dense_281 (Dense)           (None, 32)                2080

     dense_282 (Dense)           (None, 16)                528

     dropout_56 (Dropout)        (None, 16)                0

     dense_283 (Dense)           (None, 8)                 136

     batch_normalization_56 (Bat  (None, 8)                32
     chNormalization)

     dense_284 (Dense)           (None, 1)                 9

    =================================================================
    Total params: 11,041
    Trainable params: 11,025
    Non-trainable params: 16

    """
    def __init__(self, learning_rate=0.001,
                 batch_size=32, dropout_rate=0.2,
                 epochs=50, verbose= False):
        """
        Contructor of AgeRegressor class.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.verbose= verbose
    def fit(self, X, y):
        """
        Fit method. Builds the NN and fits using MAE.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Input datas on which fit will be performed.

        y : array-like of shape (n_samples)
            Labels for supervised learning.

        """
        inputs = Input(shape=X.shape[1])
        hidden = Dense(64, activation="relu")(inputs)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = Dense(16, activation="relu")(hidden)
        hidden = Dropout(self.dropout_rate)(hidden)
        hidden = Dense(8, activation="relu")(hidden)
        hidden = BatchNormalization()(hidden)
        outputs = Dense(1, activation="linear")(hidden)

        self.model = Model(inputs=inputs, outputs=outputs)
        # Compile model
        self.model.compile(
            loss="mean_absolute_error",
            optimizer="adam",
            metrics=["MAE"]
        )
        if self.verbose:
            self.model.summary()

        callbacks = [EarlyStopping(monitor="MAE",
                                   patience=10,
                                   verbose=1),
                     ReduceLROnPlateau(monitor='MAE',
                                       factor=0.1,
                                       patience=2,
                                       verbose=1)]

        history = self.model.fit(X,
                                 y,
                epochs=self.epochs,
                callbacks=callbacks,
                batch_size=self.batch_size,
                verbose=0)

        print(history.history.keys())
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

    def predict(self, X):
        """
        Predict method. Makes prediction on test data.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Input datas on which prediction will be performed..

        Returns
        -------

        self.model.predict : ndarray of shape (n_samples,) or (n_samples, n_outputs)

        """
        return self.model.predict(X)
