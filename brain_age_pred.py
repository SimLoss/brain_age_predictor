"""
This module provides functions to optimize the hyperparameters using GridSearchCV and compare them.
Best estimators found are saved in local and used to makeprediction of age on a dataframe.

Workflow:
1. Read the dataframe and make some preprocessing.
2. Split dataframe into cases and controls.
3. Split controls (CTR) dataframe in train and test set.
4. Grid search cross validation of hyperparameters of various models
    on CTR train set.
5. K-fold cross validation of optimized models on CTR train set.
   Best models setting will be saved in "best estimator" folder.
6. Best models are used to predict age on  CTR train and test set and on ASD dataset.

For each dataset, all plots will be saved in "images" folder.
"""

import os
import warnings
import pickle
from time import perf_counter

import numpy as np
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_regression, r_regression

from preprocess import *

#MODELS
models = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Regressor": RandomForestRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(max_iter=-1),
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

def cv_kfold(x, y, model, n_splits, shuffle= False, verbose= False):
    """Fit the model and make prediction using k-fold cross validation to split
    data in train/test sets. Returns the fitted model as well as the
    metrics mean values .

    Parameters
    ----------
    X : array-like
        Training data.
    y : array-like
        The target variable for supervised learning problems.
    n_splits : type
        Number of folds.
    model : object-like
        Model to be trained.
    shuffle : boolean, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    verbose : boolean, default=False
        Verbosity state. If True, it shows the model parameters after
        cross validation.
    Returns
    -------
    model: object-like
           Model fitted in cross validation.

    MAE: Numpy array
        Array containing the mean absolute error obtained in cross validation for
        each model at each iteration fold.

    MSE: Numpy array
        Array containing the mean squared error obtained in cross validation for
        each model at each iteration fold.
    PR: Numpy array
        Array containing the pearson coefficient obtained in cross validation for
        each model at each iteration fold.
    """
    try:
        x = x.to_numpy()
        y = y.to_numpy()
    except AttributeError:
        pass
    MSE = np.array([])
    MAE = np.array([])
    PR = np.array([])
    cv = KFold(n_splits, shuffle= shuffle)
    for train_index, test_index in cv.split(x):
        model_fit = model.fit(x[train_index], y[train_index])
        predict_y = model_fit.predict(x[test_index])
        #Print the model parameters.
        if verbose:
            print("Model parameters:", model.get_params())

        MSE = np.append(MSE, mean_squared_error(y[test_index], predict_y))
        MAE = np.append(MAE, mean_absolute_error(y[test_index], predict_y))
        PR = np.append(PR, pearsonr(y[test_index], predict_y))

    print(f"\nMetrics scores (mean values) on train Cross-Validation:")
    print(f"MSE:{np.mean(MSE):.3f} \u00B1 {np.std(MSE)} [years^2]")
    print(f"MAE:{np.mean(MAE):.3f} \u00B1 {np.std(MAE)} [years]")
    print(f"PR:{np.mean(PR):.3f} \u00B1 {np.std(PR)}")
    return model, MSE, MAE, PR

def model_hyp_tuner(dataframe, model, hyparams, model_name):
    """

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
    """
    x_train, y_train = drop_confounders(dataframe)[0], dataframe['AGE_AT_SCAN']

    pipe = Pipeline(
    steps=[
        ("Feature", SelectKBest()),
        ("Normalizer", Normalizer(norm='max')),
        ("Scaler", StandardScaler()),
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

    model_fit, MSE, MAE, PR = cv_kfold(x_train, y_train,
                                       model_cv.best_estimator_, 10
                                      )
    with open(
        f'best_estimator/{model_name}.pkl', 'wb'
    ) as file:
        pickle.dump(model_fit, file)

def make_predict(dataframe, model_name):
    """

    Parameters
    ----------

    dataframe : pandas dataframe
                Input dataframe to make predictions on.
    model_name : string
                Name of the used model.
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
    with open(
        "best_estimator/%s.pkl" % (model_name), "rb"
    ) as file:
        model_fit = pickle.load(file)

    x_test = drop_confounders(dataframe)[0]
    y_test = dataframe['AGE_AT_SCAN']
    age_predicted = model_fit.predict(x_test.values)
    score_metrics = {
                    "MSE": round(mean_squared_error(y_test,
                                                    age_predicted),
                                3),
                    "MAE": round(mean_absolute_error(y_test,
                                                    age_predicted),
                                3),
                    "PR": round(pearsonr(y_test,
                                        age_predicted)[0],
                                3)
                    }
    return age_predicted, y_test, score_metrics


def plot_scores(y_test, age_predicted, model, metrics,
                model_name="Regressor model", dataframe_name="Dataset metrics"):
    """
    Plots the results of the predictions vs ground truth with related metrics
    scores.

    Parameters
    ----------

    y_test : pandas dataframe
             Pandas dataframe column containing the ground truth age.
    age_predicted : array-like
                    Array containing the predicted age of each subject.
    model : function
            Regression model function.
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
              f" {model_name} model",
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

def compare_prediction( y_test1, predict_y1, y_test2, predict_y2, model_name)
):
    """Compare prediction performances of the same model on two different dataset.

    Parameters
    ----------
    y_test1 : array-like
        Test feature from the first data set.
    predict_y1 : array-like
        Predicted feauture from the first data set.
    y_test2 : array-like
        Test feature from the second data set.
    predict_y2 : type
        Predicted feature from the second data set.
    model_name : string-like
        Name of the model used for prediction.
    harmonize_option : string-like
        Harmonization method applied on data set.

    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test1, predict_y1 - y_test1, alpha=0.5, c="b", label="dataset1")
    plt.scatter(y_test2, predict_y2 - y_test2, alpha=0.5, c="g", label="dataset2")
    plt.xlabel("Ground truth Age [years]", fontsize=18)
    plt.ylabel("Delta Age [years]", fontsize=18)
    plt.title(
        "Delta Age versus Ground-truth  Age using \n \
            {}  with {} ".format(
            model_name,
            harmonize_option,
        ),
        fontsize=20,
    )
    plt.tick_params(axis="x", which="major", labelsize=18)
    plt.tick_params(axis="y", which="major", labelsize=18)
    plt.legend(loc="upper right", fontsize=14)
    plt.savefig(
        "images/delta_pred_%s.png" % (model_name),
        dpi=200,
        format="png",
    )


def CRT_ASD_split(dataframe, harm_flag=False):
    """
    Utility function to split data into 3 datasets: CTR(control) train/test and
    ASD(cases) dataset and assign them a name.

    Parameters
    ----------

    dataframe : pandas dataframe
                Input dataframe to split.

    harm_flag : boolean
                Flag indicating if the dataframe has been previously harmonized.
                DEFAULT=False.
    Returns
    -------
    df_CTR_train : pandas dataframe
                   Dataframe on which train will be performed.
                   Contains only subjects categorized as CTR cases.

    df_CTR_test : pandas dataframe
                  Dataframe on which prediction test will be performed.
                  Contains only subjects categorized as CTR cases.
    df_ASD : pandas dataframe
             Dataframe containing only subjects categorized as ASD cases.
    """
    df_ASD, df_CTR = df_split(dataframe)
    df_CTR_train, df_CTR_test = train_test_split(df_CTR,
                                                 test_size=0.3,
                                                 random_state=42)
    if harm_flag == True:
        df_CTR_train.attrs['name'] = 'df_CTR_train_Harmonized'
        df_CTR_test.attrs['name'] = 'df_CTR_test_Harmonized'
        df_ASD.attrs['name'] = 'df_ASD_Harmonized'
    else:
        df_CTR_train.attrs['name'] = 'df_CTR_train_Unharmonized'
        df_CTR_test.attrs['name'] = 'df_CTR_test_Unharmonized'
        df_ASD.attrs['name'] = 'df_ASD_Unharmonized'

    return df_CTR_train, df_CTR_test, df_ASD

#################################################MAIN
datapath='/home/cannolo/Scrivania/Università/Dispense_di_Computing/Progetto/brain_age_predictor/dataset/FS_features_ABIDE_males.csv'
#opening and setting the dataframe
df = read_df(datapath)
#adding total white matter Volume feature
add_WhiteVol_feature(df)
#removing IQ score from feature columns
df = del_FIQ(df)[0]

nharm = input("Do you want to harmonize data by provenance site using NeuroHarmonize? (yes/no)")
harm_flag = False
if nharm == "yes":
#harmonizing data by provenance site
    df = neuroharmonize(df)
    harm_flag = True

start = perf_counter()

df_CTR_train, df_CTR_test, df_ASD = CRT_ASD_split(df, harm_flag)
df_list = [df_CTR_train, df_CTR_test, df_ASD]

for model_name, model in models.items():
            model_hyp_tuner(df_CTR_train, model,
                            hyparams[model_name], model_name)

for dataframe in df_list:
            for model_name, model in models.items():
                    age_predicted, true_age, metrics= make_predict(dataframe,
                                                                    model_name)
                    pred{dataframe.attrs['name']]=[true_age, age_predicted]}
                    plot_scores(true_age, age_predicted,
                                model, metrics,
                                model_name, dataframe.attrs['name'])


stop = perf_counter()
print(f"Elapsed time {stop-start}")
