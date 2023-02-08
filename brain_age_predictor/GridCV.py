"""
Module containing functions to perform nested cross validation for
hyperparameters and parameters optimization.
"""

import numpy as np

from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os
import pickle
from time import perf_counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import f_regression

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split,
                        drop_covars,
                        add_age_class)
#setting random state for reproducibility
seed = 42

def model_tuner_cv(dataframe, model, model_name):
    """
    Create a pipeline and make (Kfold) cross-validation for hyperparameters'
    optimization.

    It makes use of the hyperparameters' grid dictionary in which, for each
    chosen model, are specified the values on which the GSCV will be performed.
    Parameters
    ----------

    dataframe : pandas dataframe
                Input dataframe containing data.

    model : function
            Regression model function.

    model_name : string
                 Name of the used model.

    Returns
    -------
    best_estimator : object-like
                     Model fitted with grid search cross validation
                     for hyperparameters.
    """

    #HYPERPARAMETER'S GRID
    hyparams = {"DDNregressor": {"Feature__k": [ 64, 128, "all"],
                                 "Feature__score_func": [f_regression],
                                 "Model__epochs": [50, 100],
                                 "Model__dropout_rate": [0.2, 0.3],
                                 #"Model__batch_size": [32, 64, -1],
                                 #"Model__learning_rate": [0.0005, 0.001, 0.0015]
                                },
                "Linear_Regression":{"Feature__k": [10, 20, 30],
                                      "Feature__score_func": [f_regression],
                                    },
                "Random_Forest_Regressor":{"Feature__k": [10, 20, 30],
                                      "Feature__score_func": [f_regression],
                                      "Model__n_estimators": [10, 100, 300],
                                      "Model__max_features": ["sqrt", "log2"],
                                      "Model__max_depth": [3, 4, 5, 6],
                                      "Model__random_state": [42],
                                          },

               "KNeighborsRegressor":{"Feature__k": [10, 20, 30],
                                      "Feature__score_func": [f_regression],
                                      "Model__n_neighbors": [5, 10, 15],
                                      "Model__weights": ['uniform','distance'],
                                      "Model__leaf_size": [20, 30, 50],
                                      "Model__p": [1,2],
                                     },
               "SVR": {"Feature__k": [10, 20, 30],
                      "Feature__score_func": [f_regression],
                      "Model__kernel": ['linear', 'poly', 'rbf'],
                      "Model__C" : [1,5,10],
                      "Model__degree" : [3,8],
                      "Model__coef0" : [0.01,10,0.5],
                      "Model__gamma" : ('auto','scale'),
                      },
               }


    x_train, y_train = drop_covars(dataframe)[0], dataframe['AGE_AT_SCAN']
    #Pipeline for setting subsequential working steps each time a model is
    #called on some data. It distinguish between train/test/val set, fitting the
    #first and only transforming the latters.
    pipe = Pipeline(
    steps=[
        ("Feature", SelectKBest()),
        ("Model", model)
        ]
    )
    print(f"\n\nOptimitazion of {model_name} parameters:")


    model_cv = GridSearchCV(
        pipe,
        cv=10,
        n_jobs=-1,
        param_grid= hyparams[model_name],
        scoring="neg_mean_absolute_error",
        verbose = 1
    )

    model_cv.fit(x_train, y_train)
    print("\nBest combination of hyperparameters:", model_cv.best_params_)
    best_estimator = model_cv.best_estimator_

    return best_estimator

def stf_kfold(dataframe, n_splits, model, model_name,
            harm_flag= False, shuffle= True, verbose= False):
    """
    Fit the model and make prediction using stratified k-fold
    cross-validation to split data in train/validation sets.
    This cross-validation object is a variation of KFold
    that returns stratified folds. The folds are made by preserving
    the percentage of samples for each class specified in "AGE_CLASS"
    column.
    Trained models are saved into "/best_estimator" folder

    Parameters
    ----------
    dataframe : pandas dataframe.
                Dataframe containin training data.

    n_splits : int
        Number of folds.

    model : object-like
        Model to be trained on cross validation.

    model_name : string
                Name of the used model.

    harm_flag : boolean, DEFAULT=False.
            Flag indicating if the dataframe has been previously harmonized.

    shuffle : boolean, DEFAULT=False.
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.

    verbose : boolean, default=False
        Verbosity state. If True, it shows the model parameters after
        cross validation.
    """
    x, y = drop_covars(dataframe)[0], dataframe['AGE_AT_SCAN']
    y_class = dataframe['AGE_CLASS']
    try:
        x = x.to_numpy()
        y = y.to_numpy()
        y_class = y_class.to_numpy()
    except AttributeError:
        pass
    #empty arrays for train metrics.
    mse_train = np.array([])
    mae_train = np.array([])
    pr_train = np.array([])

    #empty arrays for validation metrics.
    mse_val = np.array([])
    mae_val = np.array([])
    pr_val = np.array([])

    cv = StratifiedKFold(n_splits=n_splits, shuffle= shuffle, random_state=seed)
    rob_scaler = RobustScaler()

    #cross-validation
    for train_index, val_index in cv.split(x, y_class):
        model_fit = model.fit(x[train_index], y[train_index])
        predict_y_train = model_fit.predict(x[train_index])
        y[val_index] = np.squeeze(y[val_index])
        predict_y_val = model_fit.predict(x[val_index])

        mse_train = np.append(mse_train, mean_squared_error(y[train_index], predict_y_train))
        mae_train = np.append(mae_train, mean_absolute_error(y[train_index], predict_y_train))
        pr_train = np.append(pr_train, pearsonr(y[train_index], predict_y_train)[0])

        mse_val = np.append(mse_val, mean_squared_error(y[val_index], predict_y_val))
        mae_val = np.append(mae_val, mean_absolute_error(y[val_index], predict_y_val))
        pr_val = np.append(pr_val, pearsonr(y[val_index], predict_y_val)[0])

    #Print the model's parameters after cross validation.
    if verbose:
        print("Model parameters:", model.get_params())

    print("\nCross-Validation: metrics scores (mean values) on train set:")
    print(f"MSE:{np.mean(mse_train):.3f} \u00B1 {np.around(np.std(mse_train), 3)} [years^2]")
    print(f"MAE:{np.mean(mae_train):.3f} \u00B1 {np.around(np.std(mae_train), 3)} [years]")
    print(f"PR:{np.mean(pr_train):.3f} \u00B1 {np.around(np.std(pr_train), 3)}")

    print("\nCross-Validation: metrics scores (mean values) on validation set:")
    print(f"MSE:{np.mean(mse_val):.3f} \u00B1 {np.around(np.std(mse_val), 3)} [years^2]")
    print(f"MAE:{np.mean(mae_val):.3f} \u00B1 {np.around(np.std(mae_val), 3)} [years]")
    print(f"PR:{np.mean(pr_val):.3f} \u00B1 {np.around(np.std(pr_val), 3)}")

    #saving results on disk folder "../best_estimator"
    if harm_flag is True:
        saved_name = model_name + '_Harmonized'
    else:
        saved_name = model_name + '_Unharmonized'
    try:
        with open(
            f'best_estimator/grid/{saved_name}.pkl', 'wb'
        ) as file:
            pickle.dump(model_fit, file)
    except IOError:
        print("Folder \'/best_estimator\' not found.")
