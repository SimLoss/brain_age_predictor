"""
Module containing functions to perform GridSearchCV cross validation for
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import f_regression

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split,
                        drop_covars)
#setting random state for reproducibility
seed = 42

def model_tuner_cv(dataframe, model, model_name, harm_flag=False):
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
    #SCORINGS
    scorings=["neg_mean_absolute_error", "neg_mean_squared_error"]

    #HYPERPARAMETER'S GRID
    hyparams = {"DDNregressor": {"Feature__k": [ 64, 128, "all"],
                                 "Feature__score_func": [f_regression],
                                 "Model__epochs": [50, 100],
                                 "Model__dropout_rate": [0.2, 0.3],
                                 "Model__batch_size": [32, 64, -1],
                                 "Model__learning_rate": [0.0005, 0.001, 0.0015]
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
        scoring=scorings,
        refit = "neg_mean_absolute_error",
        verbose = 1,
    )

    model_cv.fit(x_train, y_train)
    print("\nBest combination of hyperparameters:", model_cv.best_params_)
    model_best = model_cv.best_estimator_

    MAE_val = np.abs(np.mean(model_cv.cv_results_["mean_test_neg_mean_absolute_error"]))
    MSE_val = np.abs(np.mean(model_cv.cv_results_["mean_test_neg_mean_squared_error"]))
    std_mae_val = np.std(model_cv.cv_results_["mean_test_neg_mean_absolute_error"])
    std_mse_val = np.std(model_cv.cv_results_["mean_test_neg_mean_squared_error"])

    print("\nCross-Validation: metrics scores (mean values) on validation set:")
    print(f"MSE:{np.around(MAE_val,3)} \u00B1 {np.around(std_mae_val,3)} [years^2]")
    print(f"MAE:{np.around(MSE_val,3)} \u00B1 {np.around(std_mse_val,3)} [years]")

    #saving results on disk folder "../best_estimator"
    if harm_flag is True:
        saved_name = model_name + '_Harmonized'
    else:
        saved_name = model_name + '_Unharmonized'
    try:
        with open(
            f'best_estimator/grid/{saved_name}.pkl', 'wb'
        ) as file:
            pickle.dump(model_best, file)
    except IOError:
        print("Folder \'/best_estimator\' not found.")
