"""
This module provides functions to optimize the hyperparameters using GridSearchCV and compares them.
Best estimators found are saved in local and used to makeprediction of age on a dataframe.

Workflow:
1. Read the dataframe and make some preprocessing.
2. Split dataframe into cases and controls.
3. Split controls (CTR) dataframe in train and test set.
4. Grid search cross validation of hyperparameters of various models
    on CTR train set.
5. K-fold cross validation of optimized models on CTR train set.
   Best models setting will be saved in "best estimator" folder.
6. Best models are used to predict age on CTR train and test set and, finally, on
   ASD dataset for a comparison of prediction between healthy subjects and the
   ones with ASD.

For each dataset, all plots will be saved in "images" folder.
"""

import os
import warnings
import pickle
from time import perf_counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import f_regression, r_regression
from denseweight import DenseWeight

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split,
                        test_scaler,
                        drop_covars)
###############################################
#MODELS
models = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(objective='reg:squarederror'),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    }
#SCORING
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']

#HYPERPARAMETER'S GRID
hyparams = {"Linear_Regression":{"Feature__k": [10, 20, 30],
                                  "Feature__score_func": [f_regression],
                                },
            "Random_Forest_Regressor":{"Feature__k": [10, 20, 30],
                                  "Feature__score_func": [f_regression],
                                  "Model__n_estimators": [10, 100, 300],
                                  "Model__max_features": ["sqrt", "log2"],
                                  "Model__max_depth": [3, 4, 5, 6],
                                  "Model__random_state": [42],
                                      },
            "XGBRegressor":{"Feature__k": [10, 20, 30],
                                  "Feature__score_func": [f_regression],
                                  "Model__n_estimators": [10, 50, 100],
                                  "Model__max_depth": [3, 6, 9],
                                  "Model__learning_rate": [0.05, 0.1, 0.20],
                                  "Model__min_child_weight": [1, 10, 100],
                                       },
           "KNeighborsRegressor":{"Feature__k": [10, 20, 30],
                                  "Feature__score_func": [f_regression],
                                  "Model__n_neighbors": [5, 10, 15],
                                  "Model__weights": ['distance'],
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
###########################################
def cv_kfold(dataframe, n_splits, model, model_name,
            harm_flag= False, shuffle= False, verbose= False):
    """
    Fit the model and make prediction using k-fold cross validation to split
    data in train/validation sets. Returns the fitted model as well as the
    metrics mean values .

    Parameters
    ----------
    dataframe : pandas dataframe
                Dataframe containin training data.

    n_splits : type
        Number of folds.

    model : object-like
        Model to be trained on cross validation.
    model_name : string
                Name of the used model.

    harm_flag : boolean
            Flag indicating if the dataframe has been previously harmonized.
            DEFAULT=False.

    shuffle : boolean, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.

    verbose : boolean, default=False
        Verbosity state. If True, it shows the model parameters after
        cross validation.
    """
    x, y = drop_covars(dataframe)[0], dataframe['AGE_AT_SCAN']

    try:
        x = x.to_numpy()
        y = y.to_numpy()
    except AttributeError:
        pass

    MSE_train = np.array([])
    MAE_train = np.array([])
    PR_train = np.array([])

    MSE_val = np.array([])
    MAE_val = np.array([])
    PR_val = np.array([])

    cv = KFold(n_splits, shuffle= shuffle)

    #cross-validation
    for train_index, val_index in cv.split(x):
        model_fit = model.fit(x[train_index], y[train_index])
        predict_y_train = model_fit.predict(x[train_index])
        predict_y_val = model_fit.predict(x[val_index])

        MSE_train = np.append(MSE_train, mean_squared_error(y[train_index], predict_y_train))
        MAE_train = np.append(MAE_train, mean_absolute_error(y[train_index], predict_y_train))
        PR_train = np.append(PR_train, pearsonr(y[train_index], predict_y_train)[0])

        MSE_val = np.append(MSE_val, mean_squared_error(y[val_index], predict_y_val))
        MAE_val = np.append(MAE_val, mean_absolute_error(y[val_index], predict_y_val))
        PR_val = np.append(PR_val, pearsonr(y[val_index], predict_y_val)[0])

    #Print the model's parameters after cross validation.
    if verbose:
        print("Model parameters:", model.get_params())

    print(f"\nCross-Validation: metrics scores (mean values) on validation set:")
    print(f"MSE:{np.mean(MSE_val):.3f} \u00B1 {np.around(np.std(MSE_val), 3)} [years^2]")
    print(f"MAE:{np.mean(MAE_val):.3f} \u00B1 {np.around(np.std(MAE_val), 3)} [years]")
    print(f"PR:{np.mean(PR_val):.3f} \u00B1 {np.around(np.std(PR_val), 3)}")

    print(f"\nCross-Validation: metrics scores (mean values) on train set:")
    print(f"MSE:{np.mean(MSE_train):.3f} \u00B1 {np.around(np.std(MSE_train), 3)} [years^2]")
    print(f"MAE:{np.mean(MAE_train):.3f} \u00B1 {np.around(np.std(MAE_train), 3)} [years]")
    print(f"PR:{np.mean(PR_train):.3f} \u00B1 {np.around(np.std(PR_train), 3)}")

    #saving results on disk folder "../best_estimator"
    if harm_flag == True:
            saved_name = model_name + '_Harmonized'
    else:
        saved_name = model_name + '_Unharmonized'

    with open(
        f'best_estimator/{saved_name}.pkl', 'wb'
    ) as file:
        pickle.dump(model_fit, file)

def model_hyp_tuner(dataframe, model, hyparams, model_name):
    """
    Performs hyperparameters tuning (optimization) using GridSearchCV then fits
    the best performing model on the training set in cross validation.

    Parameters
    ----------

    dataframe : pandas dataframe
                Input dataframe containing data.

    model : function
            Regression model function.

    hyparams : dictionary-like
                Dictionary containing model's name as key and a list of
                hyperparameters as value.
    model_name : string
                 Name of the used model.

    Returns
    -------
    best_estimator : object-like
                     Model fitted with grid search cross validation
                     for hyperparameters.
    """
    x_train, y_train = drop_covars(dataframe)[0], dataframe['AGE_AT_SCAN']

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
        param_grid=hyparams,
        scoring="neg_mean_absolute_error",
        verbose = 1
    )

    model_cv.fit(x_train, y_train)
    print("\nBest combination of hyperparameters:", model_cv.best_params_)

    best_estimator = model_cv.best_estimator_

    return best_estimator

def make_predict(dataframe, model_name, harm_flag=False):
    """

    Parameters
    ----------

    dataframe : pandas dataframe
                Input dataframe to make predictions on.

    model_name : string
                Name of the chosen model.

    harm_flag : boolean
                Flag indicating if the dataframe has been previously harmonized.
                DEFAULT=False.
    Returns
   -------
    age_predicted:  array-like
                    Array containing the predicted age of each subject.

    y_test : pandas dataframe
             Pandas dataframe column containing the ground truth age.

    score_metrics : dictionary
                    Dictionary containing names of metrics as keys and result metrics
                    for a specific model as values.
    """
    if harm_flag == True:
        saved_name = model_name + '_Harmonized'
    else:
        saved_name = model_name + '_Unharmonized'

    with open(
        "best_estimator/%s.pkl" % (saved_name), "rb"
    ) as file:
        model_fit = pickle.load(file)

    x_test = drop_covars(dataframe)[0]
    y_test = dataframe['AGE_AT_SCAN']
    age_predicted = model_fit.predict(x_test.values)
    score_metrics = {
                    "MSE": round(mean_squared_error(y_test,
                                                    age_predicted),
                                3),
                    "MAE": round(mean_absolute_error(y_test,
                                                    age_predicted),
                                3),
                    "PR":  np.around(pearsonr(y_test,
                                        age_predicted)[0],
                                3)
                    }
    return age_predicted, y_test, score_metrics


def plot_scores(y_test, age_predicted, metrics,
                model_name="Regressor model",
                dataframe_name="Dataset metrics"):
    """
    Plots the results of the predictions vs ground truth with related metrics
    scores.

    Parameters
    ----------

    y_test : pandas dataframe
             Pandas dataframe column containing the ground truth age.

    age_predicted : array-like
                    Array containing the predicted age of each subject.

    metrics : dictionary
            Dictionary containing names of metrics as keys and result metrics .
            for a specific model as values.

    model_name : string
                Model's name, DEFAULT="Regressor Model"

    dataframe_name : string
                Dataframe's name, DEFAULT="Dataset Metrics".
    """
    MSE, MAE, PR = metrics["MSE"], metrics["MAE"], metrics["PR"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, age_predicted,
               marker="*", c="r",
               label="True age"
              )
    plt.xlabel("Ground truth Age [years]", fontsize=18)
    plt.ylabel("Predicted Age [years]", fontsize=18)
    plt.plot(
        np.linspace(age_predicted.min(), age_predicted.max(), 10),
        np.linspace(age_predicted.min(), age_predicted.max(), 10),
        c="b",
        label="Prediction",
    )
    plt.title(f"Predicted vs real subject's age with"
              f" \n{model_name} model",
              fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(loc="upper right", fontsize=14)
    anchored_text = AnchoredText(f"{dataframe_name} metrics:"
                                 f"\nMAE= {MAE} [years]"
                                 f"\n MSE= {MSE} [years^2]"
                                 f"\n PR= {PR}",
                                 loc=4,
                                 borderpad=0.,
                                 frameon=True,
                                 prop=dict(fontweight="bold"),
                                )
    ax.add_artist(anchored_text)

    plt.savefig(
        "images/%s_%s.png"
        % (dataframe_name, model_name),
        dpi=200,
        format="png",
        bbox_inches="tight",
    )

    plt.show()

def delta_age( true_age1, pred_age1, true_age2, pred_age2, model_name):
    """
    Computes the difference(delta) between predicted age find with a
    specific model and true age on control test and cases dataframes.

    Parametersdelta_age
    ----------
    true_age1 : array-like
        Test feature from the first dataframe.

    pred_age1 : array-like
        Predicted feauture from the first dataframe.

    true_age2 : array-like
        Test feature from the second dataframe.

    pred_age2 : array-like
        Predicted feature from the second dataframe.

    model_name : string-like
        Name of the model used for prediction.

    """
    plt.figure(figsize=(8, 8))
    plt.scatter(true_age1, pred_age1 - true_age1, c="b", label="Control")
    plt.scatter(true_age2, pred_age2 - true_age2, alpha=0.5, c="g", label="ASD")

    plt.axhline(
        y=(pred_age1 - true_age1).mean(),
        alpha=0.5,
        color='r',
        linestyle='-',
        label=f"Δ CTR mean:{round((pred_age1 - true_age1).mean(),3)}",
    )
    plt.axhline(
        y=(pred_age2 - true_age2).mean(),
        alpha=0.5,
        color='b',
        linestyle='-',
        label=f"Δ ASD mean:{round((pred_age2 - true_age2).mean(),3)}",
    )
    plt.xlabel("Ground truth Age [years]", fontsize=18)
    plt.ylabel("Delta Age [years]", fontsize=18)
    plt.title(
        f"Delta age versus ground truth age with \n{model_name}",
        fontsize=20,)
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.legend(loc="upper right", fontsize=14)
    plt.savefig(
        "images/delta_pred_%s.png" % (model_name),
        dpi=200,
        format="png")

    plt.show()

################################################# MAIN
if __name__ == '__main__':
    datapath='/home/cannolo/Scrivania/Università/Dispense_di_Computing/Progetto/brain_age_predictor_main/brain_age_predictor/dataset/FS_features_ABIDE_males.csv'
    #opening and setting the dataframe
    df = read_df(datapath)

    #removing subject with age>40 as they're poorly represented
    df = df[df.AGE_AT_SCAN<40]

    #adding total white matter Volume feature
    add_WhiteVol_feature(df)

    harm_flag = input("Do you want to harmonize data by provenance site using NeuroHarmonize? (yes/no)")
    #harm_flag = False
    if harm_flag == "yes":
    #harmonizing data by provenance site
        df = neuroharmonize(df)
        harm_flag = True

    #splitting the dataset into ASD and CTR groups.
    ASD, CTR = df_split(df)
    #split CTR dataset into train and test.
    CTR_train, CTR_test = train_test_split(CTR,
                                           test_size=0.3,
                                           random_state=42)

    scaler = StandardScaler()
    #normalizing train set; using fit_transform.
    drop_train, drop_list = drop_covars(CTR_train)
    df_CTR_train = pd.DataFrame(scaler.fit_transform(drop_train),
                          columns = drop_train.columns, index = drop_train.index
                          )
    for column in drop_list:
        df_CTR_train[column] = CTR_train[column].values

    if harm_flag == True:
        df_CTR_train.attrs['name'] = 'df_CTR_train_Harmonized'
    else:
        df_CTR_train.attrs['name'] = 'df_CTR_train_Unharmonized'

    #Once the train set has been normalized,
    #scaling will be performed on the other datasets.
    #Only the scaler transform will be performed to avoid data leakage.
    df_CTR_test = test_scaler(CTR_test, scaler, harm_flag, "df_CTR_test")
    df_ASD = test_scaler(ASD, scaler, harm_flag, "df_ASD")

    #Performing GridSearchCV on each model followed by CV
    start = perf_counter()
    for model_name, model in models.items():
                best_estimator =model_hyp_tuner(df_CTR_train, model,
                                                hyparams[model_name],
                                                model_name)

                #now that the model is tuned, CV will be performed
                cv_kfold(df_CTR_train,
                        10,
                        best_estimator,
                        model_name,
                        harm_flag)

    stop = perf_counter()
    print(f"Elapsed time for model tuning and CV: {stop-start} s")

    df_list = [df_CTR_train, df_CTR_test, df_ASD]
    #computing predictions with fitted models then plotting results
    pred = {}
    for model_name, model in models.items():

                for dataframe in df_list:
                        age_predicted, true_age, metrics= make_predict(dataframe,
                                                                        model_name,
                                                                        harm_flag)

                        pred[dataframe.attrs['name']]=[true_age, age_predicted]
                        plot_scores(true_age, age_predicted, metrics,
                                    model_name, dataframe.attrs['name'])

                if harm_flag == True:
                    delta_age(
                        pred['df_CTR_test_Harmonized'][0],
                        pred['df_CTR_test_Harmonized'][1],
                        pred['df_ASD_Harmonized'][0],
                        pred['df_ASD_Harmonized'][1],
                        model_name
                        )
                else:
                    delta_age(
                        pred['df_CTR_test_Unharmonized'][0],
                        pred['df_CTR_test_Unharmonized'][1],
                        pred['df_ASD_Unharmonized'][0],
                        pred['df_ASD_Unharmonized'][1],
                        model_name
                        )
