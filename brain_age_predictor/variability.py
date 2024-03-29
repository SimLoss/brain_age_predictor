# pylint: disable= import-error, unnecessary-comprehension, invalid-name
# pylint: disable= C3001,

"""
Module testing the reproducibility of the results on each site's dataframe.
If no argument is stated from command line, the program will be executed without
data harmonization.
The analysis will be performed on control subjects.
"""
import sys
import argparse
import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from prettytable import PrettyTable

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split)
from brain_age_pred import make_predict
from predict_helper import plot_scores
from DDNregressor import AgeRegressor


###################################################
#MODELS
models = ["DDNregressor",
          "Linear_Regression",
          "Random_Forest_Regressor",
          "KNeighborsRegressor",
          "SVR"]

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
        default= 'dataset/FS_features_ABIDE_males.csv'
        )

    parser.add_argument('-s',
                        '--start',
                        help='Start script without harmonization.',
                        action="store_const",
                        const=True)

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

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

############################
    try:
        datapath = args.datapath
        df_ABIDE = read_df(datapath)
    except Exception as exc:
        raise FileNotFoundError('dataset/FS_features_ABIDE_males.csv'
                            'must be in your repository!') from exc

    #removing subject with age>40 as they're poorly represented
    df_ABIDE = df_ABIDE[df_ABIDE.AGE_AT_SCAN<40]

    #adding total white matter Volume feature
    add_WhiteVol_feature(df_ABIDE)

    if args.harmonize:
        nh_flag = args.harmonize
        df_ABIDE = neuroharmonize(df_ABIDE)
        HARM_STATUS = "NeuroHarmonized"
    else:
        nh_flag = args.harmonize
        HARM_STATUS = "Unharmonized"

    #splitting data in ASD and CTR dataframe, taking only
    #the latter for further analysis.
    ASD, df_CTR = df_split(df_ABIDE)

    #creating a list of datas' provenance site.
    site_list = df_CTR.SITE.unique()

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
                                                           nh_flag
                                                           )

            appender = lambda metric, key: metric.append(metrics[key])

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


            threads = [mae, mse, pr]
            for thr in threads:
                thr.start()

            for thread in threads:
                thr.join()

            #if verbose, plots the fit on each dataframe
            if args.verbose:
                plot_scores(true_age, age_predicted,
                            metrics, model_name,
                            dataframe.attrs['name']
                            )

        mean_s = np.mean(MAE)
        std_s = np.std(MAE)
        #printing a summarizing table with metrics per site
        print(f"Metrics for each site with {HARM_STATUS} dataset using {model_name} :")
        table = PrettyTable(["Metrics"]+[x for x in site_list])
        table.add_row(["MAE"]+[x for x in MAE])
        table.add_row(["MSE"] + [x for x in MSE])
        table.add_row(["PR"] + [x for x in PR])

        data_table = table.get_string()
        with open( f'metrics/grid/metrics_{model_name}_{HARM_STATUS}.txt',
                   'w', encoding="utf-8") as file:
            file.write(data_table)

        print(table)

        #making a comparative bar plot of MAE for site
        fig, ax = plt.subplots(figsize=(22, 16))
        bars = plt.bar(site_list, MAE)
        ax.bar_label(bars, fontsize=16)
        plt.xlabel("Sites", fontsize=20)
        plt.ylabel("Mean Absolute Error", fontsize=20)
        plt.title(f"MAE using {model_name} of {HARM_STATUS} sites' data ",
                  fontsize = 20)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18, rotation=50)
        anchored_text = AnchoredText(f"MAE:{np.mean(MAE):.1f} \u00B1 {np.std(MAE):.1f} [years]",
                                     loc=1,
                                     prop=dict(fontweight="bold", size=20),
                                     borderpad=0.,
                                     frameon=True,
                                    )
        ax.add_artist(anchored_text)
        plt.savefig(f"images_SITE/grid/ Sites {HARM_STATUS} with {model_name}.png",
            dpi=300,
            format="png",
            bbox_inches="tight"
            )
        plt.show()
