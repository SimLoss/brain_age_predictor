# pylint: disable=locally-disabled, import-error, too-many-arguments, invalid-name

"""
Module for Deep Dense Network implementation.
"""

import os

import absl.logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from sklearn.base import BaseEstimator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#setting seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
#clearing previous keras sessions
tf.keras.backend.clear_session()
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
     Layer (type)                Output Shape              Param #
    =================================================================
     input_1 (InputLayer)        [(None, 420)]             0

     dense (Dense)               (None, 128)               53888

     dense_1 (Dense)             (None, 64)                8256

     dense_2 (Dense)             (None, 64)                4160

     dense_3 (Dense)             (None, 32)                2080

     dropout (Dropout)           (None, 32)                0

     dense_4 (Dense)             (None, 16)                528

     dense_5 (Dense)             (None, 1)                 17

    =================================================================
    Total params: 68,929
    Trainable params: 68,929
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

    def fit(self, X, y):
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

        callbacks = [EarlyStopping(monitor="val_MAE",
                                   patience=10,
                                   verbose=1),
                    ReduceLROnPlateau(monitor='val_MAE',
                                      factor=0.1,
                                      patience=5,
                                      verbose=1)
                    ]

        history = self.model.fit(X,
                                 y,
                validation_split=0.2,
                epochs=self.epochs,
                callbacks=callbacks,
                batch_size=self.batch_size,
                verbose=1)

        if self.verbose:
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend(['train', 'validation'], loc='upper right')
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
