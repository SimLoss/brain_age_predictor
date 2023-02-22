# pylint: disable=locally-disabled, import-error, too-many-arguments, invalid-name

"""
Module for Deep Dense Network implementation.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from sklearn.base import BaseEstimator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#setting seed for reproducibility
tf.keras.utils.set_random_seed(42)
np.random.seed(42)
#clearing previous keras sessions
tf.keras.backend.clear_session()
#setting tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##############################
class AgeRegressor(BaseEstimator):
    """
    Class describing a Deep Dense Network used as a linear regressor.

    The class inherits from BaseEstimator for an easy implementation into
    scikit's pipeline and grid search.
    Linear regression is performed using 'mean absolute error' as loss func
    to minimize.

    Parameters
    ----------

    dropout_rate: float
                  Dropout rate value to be passed to dropout layer.

    epochs: int
            Number of iterations on entire dataset.

    verbose: bool, DEFAULT=False
             If True, prints the model's summary.

    Attributes
    ----------
    model : object
            Compiled model.

    Examples
    --------
    _________________________________________________________________

     Layer
      (type)                Output Shape              Param #
    _________________________________________________________________

     input_1 (InputLayer)        [(None, 64)]              0

     dense (Dense)               (None, 128)               8320

     dense_1 (Dense)             (None, 64)                8256

     dense_2 (Dense)             (None, 32)                2080

     dropout (Dropout)           (None, 32)                0

     dense_3 (Dense)             (None, 16)                528

     dense_4 (Dense)             (None, 1)                 17

    _________________________________________________________________
    Total params: 19,201
    Trainable params: 19,201
    Non-trainable params: 0
    _________________________________________________________________

    """
    def __init__(self, learning_rate=0.001,
                 batch_size=32, dropout_rate=0.2,
                 epochs=100, verbose= False):
        """
        Contructor of AgeRegressor class.
        """
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.verbose= verbose

    def fit(self, X, y, call_backs=None, val_data=None):
        """
        Fit method. Builds the NN and fits using MAE.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Input datas on which fit will be performed.

        y : array of shape (n_samples,)
            Array of labels used in train/validation.

        """
        inputs = Input(shape=X.shape[1])
        hidden = Dense(128, activation="relu")(inputs)
        hidden = Dense(64, activation="relu")(hidden)
        hidden = Dense(32, activation="relu")(hidden)
        hidden = Dropout(self.dropout_rate)(hidden)
        hidden = Dense(16, activation="relu")(hidden)
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

        history = self.model.fit(X,
                                 y,
                                 validation_data=val_data,
                                 epochs=self.epochs,
                                 callbacks=call_backs,
                                 batch_size=self.batch_size,
                                 verbose=1)

        if (self.verbose and val_data is not None):
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title('DDN Regressor loss')
            plt.ylabel('MAE [years]')
            plt.xlabel('Epochs')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.show()

        return self.model

    def predict(self, X):
        """
        Predict method. Makes prediction on test data.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Input datas on which prediction will be performed..

        Returns
        -------

        self.model.predict : array of shape (n_samples,)
                            Model prediction.
        """
        return self.model.predict(X)
