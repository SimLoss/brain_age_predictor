# pylint: disable= import-error, too-many-arguments, too-many-arguments, invalid-name
#

"""
Main module in which different models are being compared on ABIDE dataset.
Training, fit and prediction will be performed using datas from a specific
site as test set, the others as train.
User must specify if harmonization by provenance site should be performed,
using the proper command from terminal(see helper). If nothing's being stated,
harmonization won't be performed.


Workflow:
1. Read the ABIDE dataframe and make some preprocessing.
2. Split dataframe into cases and controls.
3. Splitting CTR set into train/test; choosing a single site as test.
4. Cross validation on training set.
5. Predict on site test set.
6. Repeat for another site.

For each splitting, all plots will be saved in "images_SITE" folder.
"""
import os
import sys
import argparse
from time import perf_counter

import tensorflow
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

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        neuroharmonize,
                        df_split,
                        drop_covars,
                        test_scaler,
                        train_scaler)
from DDNregressor import AgeRegressor

#setting seed for reproducibility
SEED = 42
np.random.seed(SEED)
#shutting down annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#SCORINGS
scorings=["neg_mean_absolute_error", "neg_mean_squared_error"]

#MODELS
models = {
    "DDNregressor": AgeRegressor(verbose=False),
    "Linear_Regression": LinearRegression(),
    "Random_Forest_Regressor": RandomForestRegressor(random_state=SEED),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    }


def plot_scores(true_age,
                age_predicted,
                metrics,
                model_name,
                site_name,
                dataframe_name
                ):
    """
    Plots the results of the predictions vs ground truth with related metrics
    scores.

    Parameters
    ----------

    true_age : pandas dataframe
             Pandas dataframe column containing the ground truth age.

    age_predicted : array-like
                    Array containing the predicted age of each subject.

    metrics : dictionary
             Dictionary containing names of metrics as keys and result metrics .
             for a specific model as values.

    model_name : string
                Model's name.

    site_name : string
               Site's name.

    dataframe_name : string
                    Dataframe's name, DEFAULT="Train dataset".
    """
    mse, mae, pr = metrics["MSE"], metrics["MAE"], metrics["PR"]

    ax = plt.subplots(figsize=(8, 8))[1]
    ax.scatter(true_age, age_predicted,
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
    plt.title(f"Predictions using {model_name} model",
              fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    anchored_text = AnchoredText(f"Test set: {site_name}"
                                 f"\n{dataframe_name} results:"
                                 f"\n MAE= {mae} [years]"
                                 f"\n MSE= {mse} [years^2]"
                                 f"\n PR= {pr}",
                                 loc=4,
                                 borderpad=0.,
                                 frameon=True,
                                 prop=dict(fontweight="bold"),
                                )
    ax.add_artist(anchored_text)

    plt.savefig(
                f"images_SITE/{dataframe_name}_{model_name}_{site_name}.png",
                dpi=200,
                format="png",
                bbox_inches="tight",
                )


def predict_on_site(x_pred,
                    y_pred,
                    model,
                    site_name,
                    model_name,
                    harm_flag):
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

    harm_flag : boolean.
                Flag indicating if the dataframe has been previously harmonized.
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

    if harm_flag is True:
        header = "MSE\t" + "MAE\t" + "PR\t"
        metrics = np.array([score_metrics['MSE'],
                            score_metrics['MAE'],
                            score_metrics['PR']])
        metrics = np.array(metrics).T
        np.savetxt( f"metrics/site/{site_name}_{model_name}_Harmonized.txt",
                    metrics,
                    header=header)
    else:
        header = "MSE\t" + "MAE\t" + "PR\t"
        metrics = np.array([score_metrics['MSE'],
                            score_metrics['MAE'],
                            score_metrics['PR']])
        metrics = np.array(metrics).T
        np.savetxt( f"metrics/site/{site_name}_{model_name}_Unharmonized.txt",
                    metrics,
                    header=header)

    return age_predicted, score_metrics

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

    #read dataset from path
    datapath = args.datapath
    df = read_df(datapath)

    #removing subject with age>40 as they're poorly represented
    df = df[df.AGE_AT_SCAN<40]

    #adding total white matter Volume feature
    add_WhiteVol_feature(df)

    if args.harmonize:
        nh_flag = args.harmonize
        df = neuroharmonize(df)
    else:
        nh_flag = args.harmonize
    start_time = perf_counter()
    #splitting the dataset into ASD and CTR groups.
    ASD, CTR = df_split(df)
    #creating a list of sites'names
    site_list = CTR.SITE.unique()
    #Looping on models and sites for fitting and evaluating the scores
    for site in site_list:
        #split CTR dataset into train and test: one site will be used as test, the
        #rest as training
        for name_model, regressor in models.items():
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
            #initializing metrics arrays for validation scores
            mse_val = np.array([])
            mae_val = np.array([])
            pr_val = np.array([])

            #K-fold cross-validation
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
            #validation metrics
            print(f"\nCross-Validation: {name_model} metric scores on validation set.")
            print(f"MSE:{np.mean(mse_val):.3f} \u00B1 {np.around(np.std(mse_val), 3)} [years^2]")
            print(f"MAE:{np.mean(mae_val):.3f} \u00B1 {np.around(np.std(mae_val), 3)} [years]")
            print(f"PR:{np.mean(pr_val):.3f} \u00B1 {np.around(np.std(pr_val), 3)}")

            #make prediction
            age_predicted_train, score_metrics_train = predict_on_site(x,
                                                                       y,
                                                                       model_fit,
                                                                       site,
                                                                       name_model,
                                                                       nh_flag
                                                                       )
            #plot results
            plot_scores(y,
                        age_predicted_train,
                        score_metrics_train,
                        name_model,
                        site,
                        CTR_train.attrs['name'],
                        )

            age_predicted_test, score_metrics_test = predict_on_site(x_test,
                                                                     y_test,
                                                                     model_fit,
                                                                     site,
                                                                     name_model,
                                                                     nh_flag
                                                                     )
            #plot results
            plot_scores(y_test,
                        age_predicted_test,
                        score_metrics_test,
                        name_model,
                        site,
                        CTR_test.attrs['name'],
                        )

    end_time = perf_counter()
    print(f"Elapsed time for prediction: {end_time-start_time}")