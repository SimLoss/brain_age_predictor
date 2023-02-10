"""Module implementing Leave-One-Site-Out cross validation"""
import pickle

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from preprocess import drop_covars

def losocv(dataframe, model, model_name,
           harm_flag=False, verbose=False):
    """
    Apply the LOSO cross-validation to a dataframe.

    It uses one site dataframe as test set and the others as train
    by encoding site information as labels.

    Parameters
    ----------
    dataframe : pandas dataframe.
                Dataframe containin training data.

    model : object-like
        Model to be trained on cross validation.

    model_name : string
                Name of the used model.

    harm_flag : boolean, DEFAULT=False.
            Flag indicating if the dataframe has been previously harmonized.

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
    #encoding 'SITE' feature column numerically
    dataframe.SITE = dataframe.SITE.astype('category').cat.codes
    site_label = dataframe.SITE.to_numpy()

    #empty arrays for validation metrics.
    mse_val = np.array([])
    mae_val = np.array([])
    pr_val = np.array([])


    #cross-validation with Leave-One-Group-Out; data grouped by site
    logocv = LeaveOneGroupOut()

    for train_index, val_index in logocv.split(x, y, site_label):
        model_fit = model.fit(x[train_index], y[train_index])
        predict_y_train = model_fit.predict(x[train_index])
        y[val_index] = np.squeeze(y[val_index])
        predict_y_val = model_fit.predict(x[val_index])

        mse_val = np.append(mse_val, mean_squared_error(y[val_index], predict_y_val))
        mae_val = np.append(mae_val, mean_absolute_error(y[val_index], predict_y_val))
        pr_val = np.append(pr_val, pearsonr(y[val_index], predict_y_val)[0])

        #Print the model's parameters after cross validation.
    if verbose:
        print("Model parameters:", model.get_params())

    print("\nCross-Validation: metrics scores (mean values) on validation set:")
    print(f"MSE:{np.mean(mse_val):.3f} \u00B1 {np.around(np.std(mse_val), 3)} [years^2]")
    print(f"MAE:{np.mean(mae_val):.3f} \u00B1 {np.around(np.std(mae_val), 3)} [years]")
    print(f"PR:{np.mean(pr_val):.3f} \u00B1 {np.around(np.std(pr_val), 3)}")

    #saving results on disk folder "../best_estimator"
    if harm_flag is True:
        saved_name = model_name + '_Harmonized'
    else:
        saved_name = model_name + '_Unharmonized'
    try:
        with open(
            f'best_estimator/loso/{saved_name}.pkl', 'wb'
        ) as file:
            pickle.dump(model_fit, file)
    except IOError:
        print("Folder \'/best_estimator\' not found.")
