"""
Module testing the reproducibility of the results on each site's dataframe.
The analysis will be performed on control subjects.
"""
import sys
import os
import warnings
import pickle
from time import perf_counter
import argparse
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from prettytable import PrettyTable

from preprocess import (read_df,
                        drop_covars,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split,
                        test_scaler,
                        train_scaler)
from brain_age_pred import make_predict
from predict_helper import plot_scores, residual_plot
from DDNregressor import AgeRegressor

#setting seed for reproducibility
seed = 42

###################################################
#MODELS
models = {
    "DDNregressor": AgeRegressor(),
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Regressor": RandomForestRegressor(random_state=seed),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    }
########################## MAIN
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Module for testing variability of results across different sites' dataframes."

        )

    parser.add_argument(
        "-dp",
        "--datapath",
        type = str,
        help="Path to the data folder.",
        default= '/home/cannolo/Scrivania/Università/Dispense_di_Computing/Progetto/brain_age_predictor_main/brain_age_predictor/dataset/FS_features_ABIDE_males.csv'
        )

    parser.add_argument(
        "-verb",
        "--verbose",
        action = 'store_true',
        help="Whether to show plots for each site or not."
        )

    parser.add_argument(
        "-neuroharm",
        "--harmonize",
        action = 'store_true',
        help="Use NeuroHarmonize to harmonize data by provenance site."
        )

    parser.add_argument(
        "-loso",
        "--losocv",
        action = 'store_true',
        help="Use models trained with Leave-One-Site-Out CV."
        )

    parser.add_argument(
        "-grid",
        "--gridcv",
        action = 'store_true',
        help="Use models trained with GridSearchCV."
        )

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

############################
    datapath = args.datapath
    df_ABIDE = read_df(datapath)

    #removing subject with age>40 as they're poorly represented
    df_ABIDE = df_ABIDE[df_ABIDE.AGE_AT_SCAN<40]

    #adding total white matter Volume feature
    add_WhiteVol_feature(df_ABIDE)

    if args.harmonize:
        nh_flag = args.harmonize
        df_ABIDE = neuroharmonize(df_ABIDE)
        harm_status = "NeuroHarmonized"
    else:
        nh_flag = args.harmonize
        harm_status = "Unharmonized"

    #splitting data in ASD and CTR dataframe, taking only
    #the latter for further analysis.
    ASD, CTR = df_split(df_ABIDE)
    #initializing a scaler and scaling CTR set
    rob_scaler = RobustScaler()
    df_CTR = train_scaler(CTR,
                          rob_scaler,
                          nh_flag
                          )

    #creating a list of datas' provenance site.
    site_list = df_CTR.SITE.unique()

    if args.losocv:
        dir_flag = False
    if args.gridcv:
        dir_flag = True
    #initializing and filling a dictionary that will contain
    #each different dataframe based on site.
    df_dict = {}
    for site in site_list:
        df_dict[site] = df_CTR[df_CTR.SITE == f'{site}']
        df_dict[site].attrs['name'] = f'{site}'

    #nested for loop making prediction on each model and each sites' dataframe
    for model_name in models:
            MAE = []
            MSE = []
            PR = []
            for dataframe in df_dict.values():
                    age_predicted, true_age, metrics= make_predict(dataframe,
                                                                   model_name,
                                                                   nh_flag,
                                                                   dir_flag
                                                                   )

                    appender = lambda metric, key: metric.append(metrics[key])

        #            threads = [thr.Thread(target=appender, args=())]


                    mae = threading.Thread(target=appender,
                                           name='MAE',
                                           args=(MAE, 'MAE')
                                           )

                    mse = threading.Thread(target=appender,
                                           name='MSE',
                                           args=(MSE, 'MSE')
                                           )

                    pr = threading.Thread(target=appender,
                                          name='PR',
                                          args=(PR, 'PR')
                                          )

                    mae.start()
                    mse.start()
                    pr.start()

                    mae.join()
                    mse.join()
                    pr.join()

                    mean_s = np.mean(MAE)
                    std_s = np.std(MAE)


                    #if verbose, plots the fit on each dataframe
                    if args.verbose:
                        plot_scores(true_age, age_predicted,
                                    metrics, model_name,
                                    dataframe.attrs['name']
                                    )

            #printing a summarizing table with metrics per site
            print(f"Metrics for each site with {harm_status} dataset using {model_name} :")
            table = PrettyTable(["Metrics"]+[x for x in site_list])
            table.add_row(["MAE"]+[x for x in MAE])
            table.add_row(["MSE"] + [x for x in MSE])
            table.add_row(["PR"] + [x for x in PR])

            if dir_flag is True:
                data = table.get_string()
                with open( f'metrics/grid/metrics_{model_name}_{harm_status}.txt',
                           'w') as file:
                        file.write(data)

            else:
                data = table.get_string()
                with open( f'metrics/loso/metrics_{model_name}_{harm_status}.txt',
                           'w') as file:
                        file.write(data)

            print(table)
            #making a comparative bar plot of MAE for site
            fig, ax = plt.subplots(figsize=(22, 16))
            plt.bar(site_list, MAE)
            plt.xlabel("Sites", fontsize=20)
            plt.ylabel("Mean Absolute Error", fontsize=20)
            plt.title(f"MAE using {model_name} of {harm_status} sites' data ", fontsize = 20)
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18, rotation=50)
            anchored_text = AnchoredText(f"MAE:{mean_s:.3f} \u00B1 {std_s:.3f} [years]",
                                         loc=1,
                                         prop=dict(fontweight="bold", size=20),
                                         borderpad=0.,
                                         frameon=True,
                                        )
            ax.add_artist(anchored_text)
            plt.savefig(f"images/Sites {harm_status} with {model_name}.png",
                dpi=300,
                format="png",
                bbox_inches="tight"
                )
            plt.show()
