# pylint: disable=locally-disabled, line-too-long, too-many-arguments, too-many-locals
#pylint: disable=C0103
"""
Main module in which different models are being compared on ABIDE dataset using
different cross validation: nested GridSearchCV and Leave-One-Site-Out CV.
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
   the ones with ASD.
For each dataset, all plots will be saved in "images" folder.
"""
import os
import sys
import pickle
import argparse
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import f_regression

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split,
                        drop_covars,
                        add_age_class)
from GridCV import model_tuner_cv, stf_kfold
from loso_CV import losocv
from predict_helper import plot_scores, residual_plot, test_scaler

#from DDNregressor import AgeRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
seed = 42
#MODELS
models = {
    #"DDNregressor": AgeRegressor(),
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Regressor": RandomForestRegressor(random_state=seed),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    }

def make_predict(dataframe, model_name, harm_flag=False, cv_flag=False):
    """
    Loads pre-trained model to make prediction on "unseen" test datas.
    Stores the score metrics of a prediction.
    Parameters
    ----------
    dataframe : pandas dataframe
                Input dataframe to make predictions on.
    model_name : string
                Name of the chosen model.
    harm_flag : boolean, DEFAULT=False.
                Flag indicating if the dataframe has been previously harmonized.
    cv_flag : boolean, DEFAULT=False.
                Flag indicating which kind of cross validatio has been performed.
                True: GridSearCV, False: Leave-Out-Single-Site CV.
    Returns
   -------
    age_predicted:  array-like
                    Array containing the predicted age of each subject.
    y_test : pandas dataframe
             Pandas dataframe column containing the ground truth age.
    score_metrics : dictionary
                    Dictionary containing names of metrics as keys and result
                    metrics for a specific model as values.
    """
    #loading pre-trained models from their folder
    if harm_flag is True:
        saved_name = model_name + '_Harmonized'
    else:
        saved_name = model_name + '_Unharmonized'

    if cv_flag is True:
        try:
            with open(f"best_estimator/grid/{saved_name}.pkl", "rb") as file:
                model_fit = pickle.load(file)
        except FileNotFoundError:
            print("Directory or file not found. best estimator folder must contain loso and grid folders")
    else:
        try:
            with open(f"best_estimator/loso/{saved_name}.pkl", "rb") as file:
                model_fit = pickle.load(file)
        except FileNotFoundError:
            print("Directory or file not found. best estimator folder must contain loso and grid folders")

    x_test = drop_covars(dataframe)[0]
    y_test = dataframe['AGE_AT_SCAN']
    age_predicted = model_fit.predict(x_test.values)
    age_predicted = np.squeeze(age_predicted)
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
        default= '/home/cannolo/Scrivania/Università/Dispense_di_Computing/Progetto/brain_age_predictor_main/brain_age_predictor/dataset/FS_features_ABIDE_males.csv'
        )

    parser.add_argument(
        "-loso",
        "--losocv",
        action = 'store_true',
        help="Use Leave-One-Site-Out CV."
        )

    parser.add_argument(
        "-grid",
        "--gridcv",
        action = 'store_true',
        help="Use GridSearchCV nested with StratifiedKFold."
        )
    parser.add_argument(
        "-neuroharm",
        "--harmonize",
        action = 'store_true',
        help="Use NeuroHarmonize to harmonize data by provenance site."
        )
    args = parser.parse_args()

    #=============================================================
    # STEP 1: Read the ABIDE dataframe and make some preprocessing.
    #============================================================
    datapath = args.datapath
    df = read_df(datapath)

    #removing subject with age>40 as they're poorly represented
    df = df[df.AGE_AT_SCAN<40]

    #adding total white matter Volume feature
    add_WhiteVol_feature(df)
    add_age_class(df)

    if args.harmonize:
        nh_flag = args.harmonize
    else:
        nh_flag = args.harmonize
        df = neuroharmonize(df)
    #===========================================================================
    # STEP 2: Split dataset in ASD and CTR, then the latter into train/test set.
    #===========================================================================
    #splitting the dataset into ASD and CTR groups.
    ASD, CTR = df_split(df)
    #split CTR dataset into train and test.
    CTR_train, CTR_test = train_test_split(CTR,
                                           test_size=0.3,
                                           random_state=42)
    #initializing a scaler
    rob_scaler = RobustScaler()
    #scaling train set; using fit_transform.
    drop_train, drop_list = drop_covars(CTR_train)
    df_CTR_train = pd.DataFrame(rob_scaler.fit_transform(drop_train),
                          columns = drop_train.columns, index = drop_train.index
                          )

    for column in drop_list:
        df_CTR_train[column] = CTR_train[column].values

    if nh_flag is True:
        df_CTR_train.attrs['name'] = 'df_CTR_train_Harmonized'
    else:
        df_CTR_train.attrs['name'] = 'df_CTR_train_Unharmonized'

    #using fitted scaler to transform test/ASD sets
    df_CTR_test = test_scaler(CTR_test, rob_scaler, nh_flag, "df_CTR_test")
    df_ASD = test_scaler(ASD, rob_scaler, nh_flag, "df_ASD")
    #==========================================
    # STEP 3: Cross Validation on training set.
    #==========================================
    start = perf_counter()
    for name_regressor, regressor in models.items():

        if args.gridcv:
            #Performing nested CV: GridSearchCV for each model followed by CV
            dir_flag = True
            best_hyp_estimator = model_tuner_cv(df_CTR_train,
                                                regressor,
                                                name_regressor)

            #now that the model is tuned, stratified-CV will be performed
            stf_kfold(df_CTR_train,
                      10,
                      best_hyp_estimator,
                      name_regressor,
                      nh_flag,
                     )

        if args.losocv:
            #performing Leave-One-Site-Out cross validation
            dir_flag = False
            losocv(df_CTR_train,
               regressor,
               name_regressor,
              nh_flag)

    stop = perf_counter()
    print(f"Elapsed time for model tuning and CV: {stop-start} s")
    #=======================================================
    # STEP 4: Prediction on test/ASD set and results' plots.
    #=======================================================
    df_list = [df_CTR_train, df_CTR_test, df_ASD]
    #computing predictions with fitted models then plotting results
    pred = {}
    for name_regressor in models:
        for dframe in df_list:
            predicted_age, true_age, metrics_score= make_predict(dframe,
                                                                name_regressor,
                                                                nh_flag,
                                                                dir_flag)

            pred[dframe.attrs['name']]=[true_age, predicted_age]
            plot_scores(true_age, predicted_age, metrics_score,
                        name_regressor, dframe.attrs['name'])

        if nh_flag is True:
            residual_plot(
                pred['df_CTR_test_Harmonized'][0],
                pred['df_CTR_test_Harmonized'][1],
                pred['df_ASD_Harmonized'][0],
                pred['df_ASD_Harmonized'][1],
                name_regressor
                )
        else:
            residual_plot(
                pred['df_CTR_test_Unharmonized'][0],
                pred['df_CTR_test_Unharmonized'][1],
                pred['df_ASD_Unharmonized'][0],
                pred['df_ASD_Unharmonized'][1],
                name_regressor
                    )
