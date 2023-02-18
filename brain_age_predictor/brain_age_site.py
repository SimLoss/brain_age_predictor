# pylint: disable= import-error, too-many-arguments, invalid-name

"""
Main module in which different models are being compared on ABIDE dataset.
Training, fit and prediction will be performed using datas from a specific
site as test set, the others as train.
User must specify if harmonization by provenance site should be performed,
using the proper command from terminal(see helper). If nothing's being stated,
harmonization won't be performed.


Workflow:
1. Read the ABIDE dataframe and make some preprocessing.
2. Split dataframe into cases and controls and subsequently split CTR set into
   train/test.
3. Cross validation on training set.
4. Predict on site test set.

For each splitting, all plots will be saved in "images_SITE" folder. Metrics
obtained from each cross validation are stored in "/metrics/site" folder.

"""
import os
import sys
import argparse
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from prettytable import PrettyTable

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split,
                        drop_covars,
                        test_scaler,
                        train_scaler)
from predict_helper import bar_plot
from DDNregressor import AgeRegressor

#setting seed for reproducibility
SEED = 42
np.random.seed(SEED)

#SCORINGS
scorings=["neg_mean_absolute_error", "neg_mean_squared_error"]

#MODELS
models = {
    "DDNregressor": AgeRegressor(verbose=False),
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Regressor": RandomForestRegressor(n_estimators=100,
                                                    max_features='log2',
                                                    random_state=SEED),
    "KNeighborsRegressor": KNeighborsRegressor(),
     "SVR": SVR(),
    }

def predict_on_site(x_pred,
                    y_pred,
                    model,
                    site_name,
                    model_name,
                    harm_opt):
    """
    Plots the results of the predictions vs ground truth with related metrics
    scores.

    Parameters
    ----------

    x_pred : array-like of shape (n_samples,n_features)
             Array of data on which perform prediction.

    y_pred : array-like
            Array containing labels.

    site_name : string
                Site's name.

    model_name : string
                Model's name.

    harm_opt : string.
                String indicating if the dataframe has been previously harmonized.
    Returns
    -------
    age_predicted :  array-like
            Array containing the predicted age of each subject.

    score_metrics : dictionary
                    Dictionary containing names of metrics as keys and result metrics .
                    for a specific model as values.
    """
    age_predicted = model.predict(x_pred)
    age_predicted = np.squeeze(age_predicted)
    score_metrics = {
                    "MSE": round(mean_squared_error(y_pred,
                                                    age_predicted),
                                3),
                    "MAE": round(mean_absolute_error(y_pred,
                                                    age_predicted),
                                3),
                    "PR":  np.around(pearsonr(y_pred,
                                        age_predicted)[0],
                                3)
                    }
    MAE = score_metrics['MAE']

    table = PrettyTable(["Metrics"]+[site_name])
    table.add_row(["MAE"]+[score_metrics['MAE']])
    table.add_row(["MSE"] + [score_metrics['MSE']])
    table.add_row(["PR"] + [score_metrics['PR']])
    data_table = table.get_string()

    with open( f"metrics/site/{site_name}_{model_name}_{harm_opt}.txt",
               'w') as file:
        file.write(data_table)

    return age_predicted, MAE

########################## MAIN
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Main module for brain age predictor package."
        )

    parser.add_argument('-s',
                        '--start',
                        help='Start script without harmonization.',
                        action="store_const",
                        const=True)

    parser.add_argument(
        "-dp",
        "--datapath",
        type = str,
        help="Path to the data folder.",
        default='dataset/FS_features_ABIDE_males.csv'
        )

    parser.add_argument(
        "-neuroharm",
        "--harmonize",
        action='store_true',
        help="Use NeuroHarmonize to harmonize data by provenance site."
        )
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    #=============================================================
    # STEP 1: Read the ABIDE dataframe and make some preprocessing.
    #============================================================
    #read dataset from path
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
        nh_flag = args.harmonize
        HARM_STATUS = 'Harmonized'
        df = neuroharmonize(df)
    else:
        nh_flag = args.harmonize
        HARM_STATUS = 'Unharmonized'
    start_time = perf_counter()
    #=======================================================
    # STEP 2: Splitting the dataset into ASD and CTR groups.
    #=======================================================

    ASD, CTR = df_split(df)

    #creating a list of sites'names
    site_list = CTR.SITE.unique()

    #Looping on models and sites for fitting and evaluating the scores
    for name_model, regressor in models.items():
        MAE = []
        for site in site_list:
            #split CTR dataset into train and test: one site will be used as test, the
            #rest as training
            print(f"\nUsing {site} as test set.")
            CTR_test = CTR.loc[CTR['SITE'] == f'{site}']
            CTR_train = CTR.drop(CTR[CTR.SITE == f'{site}'].index, axis=0)

            #initializing a scaler and scaling train set
            rob_scaler = RobustScaler()
            CTR_train = train_scaler(CTR_train, rob_scaler, nh_flag)

            #using fitted scaler to transform test/ASD sets
            CTR_test = test_scaler(CTR_test, rob_scaler, nh_flag, "CTR_test")

            x, y = drop_covars(CTR_train)[0], CTR_train['AGE_AT_SCAN']
            x_test, y_test = drop_covars(CTR_test)[0], CTR_test['AGE_AT_SCAN']
            try:
                x = x.to_numpy()
                y = y.to_numpy()
                x_test = x_test.to_numpy()
                y_test = y_test.to_numpy()
            except AttributeError:
                pass

            #================================
            # STEP 3: KFold cross validation.
            #================================
            #initializing metrics arrays for validation scores
            mse_val = np.array([])
            mae_val = np.array([])
            pr_val = np.array([])
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            for train_index, val_index in cv.split(x, y):
                model_fit = regressor.fit(x[train_index], y[train_index])
                predict_y_train = model_fit.predict(x[train_index])
                y[val_index] = np.squeeze(y[val_index])
                predict_y_val = model_fit.predict(x[val_index])

                mse_val = np.append(mse_val, mean_squared_error(y[val_index],
                                                                predict_y_val))
                mae_val = np.append(mae_val, mean_absolute_error(y[val_index],
                                                                predict_y_val))
                pr_val = np.append(pr_val, pearsonr(y[val_index],
                                                    predict_y_val)[0])

            print(f"\nCross-Validation: {name_model} metric scores on validation set.")
            print(f"MSE:{np.mean(mse_val):.3f} \u00B1 {np.around(np.std(mse_val), 3)} [years^2]")
            print(f"MAE:{np.mean(mae_val):.3f} \u00B1 {np.around(np.std(mae_val), 3)} [years]")
            print(f"PR:{np.mean(pr_val):.3f} \u00B1 {np.around(np.std(pr_val), 3)}")

            #==================================
            # STEP 4: Predict on site test set.
            #==================================

            age_predicted_test, site_MAE = predict_on_site(x_test,
                                                             y_test,
                                                             model_fit,
                                                             site,
                                                             name_model,
                                                             HARM_STATUS
                                                             )

            MAE.append(site_MAE)

        #plot results in a summarizing barplot
        fig, ax = plt.subplots(figsize=(22, 16))
        bars = plt.bar(list_of_sites, metric)
        ax.bar_label(bars, fontsize=16)
        plt.xlabel("Sites", fontsize=20)
        plt.ylabel("Mean Absolute Error", fontsize=20)
        plt.title(f"MAE using {regressor_name} of {harm_stat} sites' data ",
                  fontsize = 20)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18, rotation=50)
        anchored_text = AnchoredText(f"MAE:{np.mean(metric):.3f} \u00B1 {np.std(metric):.3f} [years]",
                                     loc=1,
                                     prop=dict(fontweight="bold", size=20),
                                     borderpad=0.,
                                     frameon=True,
                                    )
        ax.add_artist(anchored_text)
        plt.savefig(f"images_SITE/site/Sites {harm_stat} with {regressor_name}.png",
            dpi=300,
            format="png",
            bbox_inches="tight"
            )
        plt.show()

    end_time = perf_counter()
    print(f"Elapsed time for prediction: {end_time-start_time}")
