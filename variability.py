"""
Module testing the reproducibility of the results on each site's dataframe.
It works with previously trained models and fitted scaler(from sklearn.preprocessing).
When asked, one must specify whether the scaler has been fitted on harmonized
data or not by adding  '_Harmonize' or '_Unharmonized' to scaler's name.
Custom scaler should be imported manually.
The analysis will be performed on control subjects.

"""
import os
import warnings
import pickle
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import *
from prettytable import PrettyTable

from preprocess import read_df,add_WhiteVol_feature, neuroharmonize, df_split, test_scaler
from brain_age_pred import make_predict, plot_scores
import multiprocessing as mp

#MODELS
models = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Regressor": RandomForestRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    }


##################################### PREPROCESSING
datapath='/home/cannolo/Scrivania/Università/Dispense_di_Computing/Progetto/brain_age_predictor/dataset/FS_features_ABIDE_males.csv'
#opening and setting the dataframe
df_ABIDE = read_df(datapath)

#removing subject with age>40 as they're poorly represented
df_ABIDE = df_ABIDE[df_ABIDE.AGE_AT_SCAN<40]

#adding total white matter Volume feature
add_WhiteVol_feature(df_ABIDE)

harm_flag = input("Do you want to harmonize data by provenance site using NeuroHarmonize? (yes/no)")
if harm_flag == "yes":
#harmonizing data by provenance site
    df_ABIDE = neuroharmonize(df_ABIDE)
    harm_flag = True
    harm_status = "NeuroHarmonized"
else:
    harm_status = "Normalized"

#splitting data in ASD and CTR dataframe, taking only
#the latter for further analysis.
df_ASD, df_CTR = df_split(df_ABIDE)

##################################### MAIN

#loading a pre-trained scaler. Must be coherent with
#the one chosen during train.
scaler_name = input("A pre-fitted scaler will be used for normalize data. Please, enter the name of a valid scaler(e.g : StandardScaler, RobustScaler, etc)")
with open( f"best_estimator/scaler/{scaler_name}.pkl", "rb") as file:
    scaler = pickle.load(file)
df = test_scaler(df_CTR, scaler, harm_flag)

#creating a list of datas' provenance site.
site_list = df.SITE.unique()
#
#initializing and filling a dictionary that will contain
#each different dataframe based on site.
df_dict = {}
for site in site_list:
    df_dict[site] = df[df.SITE == f'{site}']
    df_dict[site].attrs['name'] = f'{site}'

verbose = input("Do you want to display plot for each site?(yes/no)")
#nested for loop making prediction on each model and each sites' dataframe
for model_name in models.keys():
        MAE = []
        MSE = []
        PR = []
        for dataframe in df_dict.values():
                age_predicted, true_age, metrics= make_predict(dataframe,
                                                                model_name,
                                                                harm_flag)
                MAE.append(metrics['MAE'])
                MSE.append(metrics['MSE'])
                PR.append(metrics['PR'])
                mean_s = np.mean(MAE)
                std_s = np.std(MAE)
                #if verbose, plots the fit on each dataframe
                if verbose == "yes":
                    plot_scores(true_age, age_predicted,
                                metrics, model_name, dataframe.attrs['name'])
        #printing a summarizing table with metrics per site
        print(f"MAE[years] for each site with {harm_status} data using {model_name} :")
        table = PrettyTable(["Metrics"]+[x for x in site_list])
        table.add_row(["MAE"]+[x for x in MAE])
        table.add_row(["MSE"] + [x for x in MSE])
        table.add_row(["PR"] + [x for x in PR])
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
