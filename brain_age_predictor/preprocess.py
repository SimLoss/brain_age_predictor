""" This module provides some function to read, explore and preprocess an input
    dataframe cointaining a set of features. Specifically, it allows to:
    - read data from a file and build a dataframe;
    - explore data info and make some plots
    - add/remove features from the dataframe;
    - split data in cases and controls group;
    - normalize and harmonize data;

It may be also used as a standalone program to explore the dataset.
"""
import sys
import os
import logging
import inspect
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from neuroHarmonize import harmonizationLearn


def read_df(dataset_path):
    """
    Reads a .csv file from data file and returns it as a pandas dataframe.
    ID and acquisition site of each subject contained into the dataframe are
    extracted from the "FILE_ID" column. The latter is stored in a proper
    dataframe's column, while the first is used as dataframe index.

    Parameters
    ----------
        dataset_path : str
            Path to the dataset file.


    Returns
    -------
        df: pandas DataFrame
            Dataframe containing the features of each subject by rows.
    """

    try:
        logging.info("Reading dataset..")
        dataframe = pd.read_csv(dataset_path, sep = ';')
        dataframe.attrs['name'] = "Unharmonized ABIDE dataframe"
        site = []
        ind = []
        for idx in dataframe.FILE_ID:
            ind.append(idx.split('_')[-1])
            site.append(idx.split('_')[0])
        #adding site column to dataframe and using FILE_ID as index
        dataframe['SITE'] = site
        dataframe['FILE_ID'] = ind
        dataframe = dataframe.set_index('FILE_ID')
    except OSError:
            print("Invalid file or path.")
    return dataframe

def data_info(dataframe):
    """
    Shows some useful information about the feature's dataset.

    Parameters
    ----------
        dataframe: pandas DataFrame
             Dataframe containing the features of each subject


    Returns
    -------
    None
    """
    print(f"Dataframe info:")
    print(dataframe.info(memory_usage = False))
    print(f"\n\nDataframe size: {dataframe.size} elements" )
    print(f"\n\nNumber of ASD cases:"
        +f" {dataframe[dataframe.DX_GROUP == 1].AGE_AT_SCAN.count()}")
    print(f"Number of Controls:"
        +f" {dataframe[dataframe.DX_GROUP == -1].AGE_AT_SCAN.count()}")
    print(f"Mean Age in ASD set: "
        +f"{dataframe[dataframe.DX_GROUP == -1]['AGE_AT_SCAN'].values.mean()}"
        +f" \u00B1 {dataframe[dataframe.DX_GROUP == -1]['AGE_AT_SCAN'].values.std()}")
    print(f"Mean Age in CTR set: "
        +f"{dataframe[dataframe.DX_GROUP == 1]['AGE_AT_SCAN'].values.mean()}"
        +f" \u00B1 {dataframe[dataframe.DX_GROUP == 1]['AGE_AT_SCAN'].values.std()}")
    print("\n\nShowing the first and last 10 rows of the dataframe.. ")
    print(dataframe.head(10))
    print(dataframe.tail(10))

    print("\n\nShowing statistical quantities for each column after normalization..")
    normalized_df = normalization(dataframe)
    des = normalized_df.describe()
    print(des)

def df_split(dataframe):
    """
    Splits the dataframe's subjects in two different groups based on their
    clinical classification: ASD (Autism Spectre Disorder) and Controls.

    Parameters
    ----------

    dataframe : pandas DataFrame
                The dataframe of data to be split.

    Returns
    -------

    df_AS : pandas DataFrame
            Dataframe containing ASD cases.

    df_TD : pandas DataFrame
            Dataframe containing controls.

    """
    logging.info("Splitting the dataframe in cases and controls..")
    df_ASD = dataframe.loc[dataframe.DX_GROUP == 1]
    df_ASD.attrs['name'] = 'ASD'
    df_CTR = dataframe.loc[dataframe.DX_GROUP == -1]
    dataframe.attrs['name'] = 'CTR'
    return df_ASD, df_CTR

def add_WhiteVol_feature(dataframe):
    """
    Adds a column with total brain's white matter volume.

    Parameters
    ----------

    dataframe : pandas DataFrame
                Dataframe to be passed to the function.
    """

    #sum left and right hemisphere's white matter volume
    cols = ['lhCerebralWhiteMatterVol','rhCerebralWhiteMatterVol']
    dataframe['TotalWhiteVol'] = dataframe[cols].sum(axis=1)

def drop_covars(dataframe):
    """
    Drops the following columns with covariate and confounding variables from
    the dataframe: "SITE","AGE_AT_SCAN","DX_GROUP","SEX". "FIQ".

    Parameters
    ----------

    dataframe : pandas DataFrame
                Dataframe from wich will be dropped the indicated columns.

    Returns
    -------
    dataframe : pandas Dataframe
                Dataframe without the aforementioned columns.
    covar_list : list
                List of strings containing the name of the dropped columns.
    """
    covar_list = ["SITE","AGE_AT_SCAN","DX_GROUP","SEX","FIQ"]
    dataframe = dataframe.drop(covar_list, axis=1)

    return dataframe, covar_list

def plot_histogram(dataframe, feature):
    """
    Plots histogram of a given feature of the dataframe.
    Plots will be saved in data_plots folder.

    Parameters
    ----------

    dataframe : pandas DataFrame
                Dataframe to which apply hist method.
    feature : string
              Feature to plot the histogram of.
    """
    if feature == 'SITE':
        dataframe.value_counts('SITE').plot(fontsize=14, kind='bar', grid=True,)
        plt.ylabel("Subjects", fontsize=18)
        plt.xlabel(f"{feature}", fontsize=18)
        plt.title("N. subjects per provenance site", fontsize=20)
        plt.show()

    else:
        dataframe.hist([feature], figsize=(8, 8), bins=100, grid = True)
        plt.ylabel("Subjects", fontsize=18)
        plt.xlabel(f"{feature}", fontsize=18)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.title(f"Histogram of n. subjects VS {feature}", fontsize=20)
        plt.show()

    plt.savefig(f"data_plots/{feature}_histogram.png",
                dpi=300,
                format="png",
                bbox_inches="tight"
                )

def plot_box(dataframe, feat_x, feat_y):
    """
    Draw a box plot to show distributions of a feature with respect of another.
    Plots will be saved in data_plots folder.
    Parameters
    ----------

    dataframe : pandas DataFrame
                Dataframe where the specified features are taken from.
    feat_x : string
              Feature showed on the x-axis of the boxplot.
    feat_y : string
              Feature showed on the y-axis of the boxplot.
    """

    sns_boxplot = sns.boxplot(x=feat_x, y=feat_y, data=dataframe)
    labels = sns_boxplot.get_xticklabels()
    sns_boxplot.set_xticklabels(labels,rotation=50, fontsize=14)
    plt.yticks(fontsize=18)
    plt.xlabel(f"{feat_x}", fontsize=18)
    plt.ylabel(f"{feat_y}", fontsize=18)
    sns_boxplot.set_title(f"Boxplot of {dataframe.attrs['name']}",
                          fontsize=20, pad=20)
    sns_boxplot.grid()

    plt.savefig(f"data_plots/{dataframe.attrs['name']}_box plot.png",
                dpi=300,
                format="png",
                bbox_inches="tight"
                )

    plt.show()

def normalization(dataframe):
    """
    Makes data normalization by scaling each feature to (0,1) range.

    Parameters
    ----------

    dataframe : pandas DataFrame
                Dataframe to be passed to the function.
    Returns
    -------
    norm_df: pandas Dataframe
             Normalized dataframe.
    """
    scaler = MinMaxScaler()
    #scaling dataframe columns by max; using fit_transform.
    drop_df, drop_list = drop_covars(dataframe)
    norm_df = pd.DataFrame(scaler.fit_transform(drop_df),
                          columns = drop_df.columns, index = drop_df.index
                          )
    for column in drop_list:
            norm_df[column] = dataframe[column].values
    norm_df.attrs['name'] = "Normalized dataframe"

    return norm_df

def test_scaler(dataframe, scaler, harm_flag=False, dataframe_name="Dataframe"):
    """
    Utility function to normalize test set using only transform
    method from the scaler.
    Parameters
    ----------
    dataframe : pandas dataframe
                Input dataframe to be normalized.

    scaler : object-like
             Scaler used to perform dataframe normalization with transform
             method. Should be previously fitted on the train set and implement
             a transform method.

    harm_flag : boolean
                Flag indicating if the dataframe has been previously harmonized.
                DEFAULT=False.

    dataframe_name : string
                    Name of the data
    Returns
    -------
    scaled_df : pandas dataframe.
            Normalized dataframe.
    """
    drop_test, drop_list = drop_covars(dataframe)
    scaled_df = pd.DataFrame(scaler.transform(drop_test))
    scaled_df.columns = drop_test.columns
    for column in drop_list:
        scaled_df[column] = dataframe[column].values

    if harm_flag is True:
        scaled_df.attrs['name'] = f'{dataframe_name}_Harmonized'
    else:
        scaled_df.attrs['name'] = f'{dataframe_name}_Unharmonized'

    return scaled_df

def train_scaler(dataframe, scaler, harm_flag=False):
    """
    Utility function to normalize test set using only transform
    method from the scaler.
    Parameters
    ----------
    dataframe : pandas dataframe
                Input dataframe to be normalized.

    scaler : object-like
             Scaler used to perform dataframe normalization with transform
             method. Should be previously fitted on the train set and implement
             a transform method.

    harm_flag : boolean
                Flag indicating if the dataframe has been previously harmonized.
                DEFAULT=False.

    dataframe_name : string
                    Name of the data
    Returns
    -------
    scaled_df : pandas dataframe.
            Normalized dataframe.
    """

    drop_train, drop_list = drop_covars(dataframe)
    scaled_df = pd.DataFrame(scaler.fit_transform(drop_train),
                          columns = drop_train.columns, index = drop_train.index
                          )

    for column in drop_list:
        scaled_df[column] = dataframe[column].values

    if harm_flag is True:
        scaled_df.attrs['name'] = 'df_CTR_train_Harmonized'
    else:
        scaled_df.attrs['name'] = 'df_CTR_train_Unharmonized'

    return scaled_df

def neuroharmonize(dataframe, covariate= ["SITE","AGE_AT_SCAN"]):
    """
    Harmonize dataset using neuroHarmonize, a harmonization tools for
    multi-site neuroimaging analysis. Workflow:
    1-Load your data and all numeric covariates;
    2-Run harmonization and store the adjusted data.

    Parameters
    ----------

    dataframe : pandas DataFrame
                Input dataframe containing all covariates to control for
                during harmonization.
                Must have a single column named 'SITE' with labels
                that identifies sites.

    covariate : list, default=['AGE_AT_SCAN']
                List of strings that contains covariates to control for
                during harmonization.
                All covariates must be encoded numerically (no categorical
                variables) and list must contain a single column "SITE" with
                site labels.

    Returns
    -------

    df_neuro_harmonized: pandas DataFrame
                          Dataframe containing harmonized data.
    """
    #firstly, drop the covariate columns from the dataframe
    dropped_df, covar_list = drop_covars(dataframe)
    df_array = np.array(dropped_df)
    #stating the covariates (here we're using just one)
    covars = dataframe.loc[:,covariate]

    model, array_neuro_harmonized = harmonizationLearn(df_array, covars)

    df_neuro_harmonized = pd.DataFrame(array_neuro_harmonized, index=dataframe.index)
    df_neuro_harmonized.attrs['name'] = "Harmonized ABIDE"
    df_neuro_harmonized.columns = dropped_df.columns

    for column in covar_list:
        df_neuro_harmonized[column] = dataframe[column].values

    return df_neuro_harmonized

################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Data exploration module for ABIDE dataset."
    )

    parser.add_argument(
        "-dp",
        "--datapath",
        type = str,
        help="Path to the data folder.",
        default= 'dataset/FS_features_ABIDE_males.csv'

    )

    parser.add_argument(
        "-norm",
        "--normalize",
        action = 'store_true',
        help="Use NeuroHarmonize to harmonize data by provenance site."
        )

    parser.add_argument(
        "-neuroharm",
        "--harmonize",
        action = 'store_true',
        help="Use NeuroHarmonize to harmonize data by provenance site."
        )
    parser.add_argument(
        "-exp",
        "--exploration",
        action = 'store_true',
        help="Shows various informations of the dataframe.",
    )
    parser.add_argument(
        "-hist",
        "--histogram",
        type= str,
        help="Plot and save the frequency histogram of the specified feature.",
    )
    parser.add_argument(
        "-box",
        "--boxplot",
        type= str,
        nargs=2,
        help= "Draw and save a box plot to show distributions of two specified feature (e. g. feat_x feat_y). ",
    )
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

#############################################################

    if args.datapath:
        datapath = args.datapath
        dataframe = read_df(datapath)

    add_WhiteVol_feature(dataframe)

    if args. normalize:
        dataframe = normalization(dataframe)

    if args.harmonize:
        dataframe = neuroharmonize(dataframe)

    if args.exploration:
        data_info(dataframe)

    if args.histogram:
        plot_histogram(dataframe, args.histogram)

    if args.boxplot:
        plot_box(dataframe, args.boxplot[0], args.boxplot[1])
