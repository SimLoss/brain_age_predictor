import unittest
from pathlib import Path
import os
import sys
import warnings
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression

#to run locally
#package_name = "../brain_age_predictor"
package_name = "../brain_age_predictor"

sys.path.insert(0, package_name)

from preprocess import (read_df,
                        add_WhiteVol_feature,
                        df_split,
                        drop_covars,
                        normalization,
                        train_scaler,
                        neuroharmonize)
from brain_age_pred import make_predict

class TestBrainAge(unittest.TestCase):
    """
    Unit test for the microcal_classifier project.
    """
    def setUp(self):
        """
        Class setup.
        """
        self.data = package_name + "/dataset/FS_features_ABIDE_males.csv"

    def test_read_df(self):
        """
        Test for read_df function.
        """
        dataframe =  read_df(self.data)
        self.assertEqual(dataframe.size, 387960, 'Wrong Size')
        self.assertEqual(dataframe.shape, (915, 424), 'Wrong Shape')
        self.assertIn('SITE', dataframe.keys(), 'SITE was not added')

    def test_add_WhiteVol_feature(self):
        """
        Test for add_WhiteVol_feature function.
        """
        dataframe = read_df(self.data)
        add_WhiteVol_feature(dataframe)
        self.assertIn('TotalWhiteVol', dataframe.keys(),
                      'TotalWhiteVol was not added')
        self.assertEqual(dataframe.shape, (915, 425), 'Wrong Shape')

    def test_df_split(self):
        """
        Test for df_split function.
        """
        dataframe = read_df(self.data)
        df_ASD, df_CTR = df_split(dataframe)
        assert df_ASD.shape == (451, 424)
        assert df_CTR.shape == (464, 424)

    def test_drop_covars(self):
        """
        Test for drop_covars function.
        """
        dataframe = read_df(self.data)
        dataframe, covar_list = drop_covars(dataframe)
        self.assertEqual(dataframe.shape, (915, 419), 'Wrong Shape')
        self.assertNotIn('SITE', dataframe.keys(), 'SITE was not dropped')
        self.assertNotIn('AGE_AT_SCAN', dataframe.keys(),
                         'AGE_AT_SCAN was not dropped')

    def test_normalization(self):
        """
        Test for normalization function.
        """
        dataframe = read_df(self.data)
        add_WhiteVol_feature(dataframe)
        scaled_df = normalization(dataframe)
        scaled_df = drop_covars(scaled_df)[0]
        self.assertTrue(all(scaled_df['TotalWhiteVol']<=1),
                            "Dataframe was not normalized")

    def test_train_scaler(self):
        """
        Test for train_scaler function.
        """
        dataframe = read_df(self.data)
        add_WhiteVol_feature(dataframe)
        scaler = RobustScaler()
        scaled_df = train_scaler(dataframe,
                                scaler,
                                harm_flag=True)
        self.assertNotEqual(scaled_df['TotalWhiteVol'].to_numpy().all(),
                            dataframe['TotalWhiteVol'].to_numpy().all())

    def test_neuroharmonize(self):
        """
        Test for neuroharmonize function.
        """
        dataframe = read_df(self.data)
        add_WhiteVol_feature(dataframe)
        df_neuro_harmonized = neuroharmonize(dataframe)
        self.assertTrue(df_neuro_harmonized['rh_MeanThickness'].to_numpy().mean() >
                        dataframe['rh_MeanThickness'].to_numpy().mean())

if __name__ == "__main__":
    unittest.main()
