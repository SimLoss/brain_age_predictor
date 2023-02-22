# pylint: disable=locally-disabled, import-error
"""
Main module in which different models are being compared on ABIDE dataset using
GridSearchCV.
User must specify if harmonization by provenance site should be performed,
using the proper command from terminal(see helper). If nothing's being stated,
harmonization won't be performed.
Best estimators found are saved in local and used to make prediction of age
on test set.

Workflow:
    1. Read the ABIDE dataframe and make some preprocessing.
    2. Split dataframe into cases and controls, the latter (CTR) in
    train and test set.
    3. Cross validation on training set.
        Best models setting will be saved in "best estimator" folder.
    4. Best models are used to predict age on CTR train and test set and, finally,
        on ASD dataset for a comparison of prediction between healthy subjects and
        cases.

For each dataset, all plots will be saved in "images" folder.

"""
import os
import sys
import pickle
import argparse
from time import perf_counter

import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split,
                        drop_covars)
from grid_CV import model_tuner_cv
from predict_helper import plot_scores, residual_plot
from DDNregressor import AgeRegressor

tf.keras.utils.set_random_seed(42)


def make_predict(dataframe, model_name, harm_flag=False):
    """
    Loads pre-trained model to make prediction on "unseen" test datas.

    Stores the score metrics of a prediction.

    Parameters
    ----------
    dataframe : pandas dataframe.
                Input dataframe to make predictions on.

    model_name : string
                Name of the chosen model.

    harm_flag : boolean, DEFAULT=False.
                Flag indicating if the dataframe has been previously harmonized.

    Returns
    -------
    predicted_age : array-like.
                    Array containing the predicted age of each subject.

    y_test : pandas dataframe.
             Pandas dataframe column containing the ground truth age.

    score_metrics : dictionary.
                    Dictionary containing names of metrics as keys and result
                    metrics for a specific model as values.
    """
    #loading pre-trained models from their folder
    if harm_flag is True:
        saved_name = model_name + '_Harmonized'
    else:
        saved_name = model_name + '_Unharmonized'

    try:
        with open(f"best_estimator/{saved_name}.pkl", "rb") as file:
            model_fit = pickle.load(file)
    except Exception as exc:
        raise FileNotFoundError("Directory or file not found."
                +"Models must be trained first.") from exc

    #prediction
    x_test = drop_covars(dataframe)[0]
    y_test = dataframe['AGE_AT_SCAN']
    age_predicted = model_fit.predict(x_test.values)
    age_predicted = np.squeeze(age_predicted)
    score_metrics = {
                    "MSE": round(mean_squared_error(y_test,
                                                    age_predicted),
                                1),
                    "MAE": round(mean_absolute_error(y_test,
                                                    age_predicted),
                                1),
                    "PR":  np.around(pearsonr(y_test,
                                        age_predicted)[0],
                                1)
                    }
    return age_predicted, y_test, score_metrics

########################## MAIN
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Main module for brain age predictor package."
        )
    parser.add_argument(
        "-dp",
        "--datapath",
        type = str,
        help="Path to the data folder.",
        default='dataset/FS_features_ABIDE_males.csv'
        )

    parser.add_argument(
        "-grid",
        "--gridcv",
        action='store_true',
        help="Use GridSearch cross validation to train and fit models."
        )

    parser.add_argument(
        "-pred",
        "--predict",
        action='store_true',
        help="Make predictions with models pre-trained with GridSearchCV."
        )

    parser.add_argument(
        "-neuroharm",
        "--harmonize",
        action='store_true',
        help="Use NeuroHarmonize to harmonize data by provenance site."
        )

    parser.add_argument(
        "-verb",
        "--verbose",
        action='store_true',
        help="Set DDN Regressor model's verbosity. If True, it shows model summary."
            "Default = False"
        )
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    #MODELS
    models = {
        "DDNregressor": AgeRegressor(verbose=args.verbose),
        "Linear_Regression": LinearRegression(),
        "Random_Forest_Regressor": RandomForestRegressor(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "SVR": SVR(),
        }


    #=============================================================
    # STEP 1: Read the ABIDE dataframe and make some preprocessing.
    #=============================================================

    try:
        datapath = args.datapath
        df = read_df(datapath)
    except Exception as exc:
        raise FileNotFoundError('dataset/FS_features_ABIDE_males.csv'
                            'must be in your repository!') from exc


    #removing subject with age>40 as they're poorly represented
    df = df[df.AGE_AT_SCAN<40]

    #adding total white matter Volume feature
    add_WhiteVol_feature(df)

    if args.harmonize:
        NH_FLAG = args.harmonize
        df = neuroharmonize(df)
    else:
        NH_FLAG = args.harmonize

    #===========================================================================
    # STEP 2: Split dataset in ASD and CTR, then the latter into train/test set.
    #         Scaling datasets.
    #===========================================================================
    #splitting the dataset into ASD and CTR groups.
    ASD, CTR = df_split(df)
    #split CTR dataset into train and test.
    CTR_train, CTR_test = train_test_split(CTR,
                                           test_size=0.3,
                                           random_state=42)
    if NH_FLAG is True:
        CTR_train.attrs['name'] = 'df_CTR_train_Harmonized'
        CTR_test.attrs['name'] = 'df_CTR_test_Harmonized'
        ASD.attrs['name'] = 'df_ASD_Harmonized'
    else:
        CTR_train.attrs['name'] = 'df_CTR_train_Unharmonized'
        CTR_test.attrs['name'] = 'df_CTR_test_Unharmonized'
        ASD.attrs['name'] = 'df_ASD_Unharmonized'
    #===================================================================
    # STEP 3: Cross Validation on training set (only if gridcv is True).
    #===================================================================
    start = perf_counter()
    for name_regressor, regressor in models.items():

        if args.gridcv:
            #Performing GridSearch Cross Validation
            best_hyp_estimator = model_tuner_cv(CTR_train,
                                                regressor,
                                                name_regressor,
                                                NH_FLAG)

        if not (args.gridcv or args.predict):
            parser.error('Required one of the following arguments: -grid, -pred')

    stop = perf_counter()
    print(f"Elapsed time for model tuning and CV: {stop-start} s")

    #=======================================================
    # STEP 4: Prediction on test/ASD set and results' plots.
    #=======================================================
    df_list = [CTR_train, CTR_test, ASD]

    #make prediction and plot scores
    pred = {}
    for name_regressor in models:
        for dframe in df_list:
            predicted_age, true_age, metrics_score= make_predict(dframe,
                                                                name_regressor,
                                                                NH_FLAG)

            pred[dframe.attrs['name']]=[true_age, predicted_age]

            plot_scores(true_age,
                        predicted_age,
                        metrics_score,
                        name_regressor,
                        dframe.attrs['name'],
                        )

        if NH_FLAG is True:
            residual_plot(pred['df_CTR_test_Harmonized'][0],
                          pred['df_CTR_test_Harmonized'][1],
                          pred['df_ASD_Harmonized'][0],
                          pred['df_ASD_Harmonized'][1],
                          name_regressor,
                          NH_FLAG)
        else:
            residual_plot(pred['df_CTR_test_Unharmonized'][0],
                          pred['df_CTR_test_Unharmonized'][1],
                          pred['df_ASD_Unharmonized'][0],
                          pred['df_ASD_Unharmonized'][1],
                          name_regressor,
                          NH_FLAG)
