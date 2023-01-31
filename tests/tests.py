import unittest
from pathlib import Path
import os
import sys


#sys.path.insert(0, str(Path(os.getcwd()).parent)) # Get the absolute path to the parent dir.
#package_name = "../microcal_classifier/"
#sys.path.insert(0, package_name)
#to run locally
#package_name = "../BrainAge"
package_name = "../brain_age_predictor"

sys.path.insert(0, package_name)

from preprocess import read_df, add_WhiteVol_feature, df_split, drop_covars, normalization
#from brain_age_pred import *

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
        dataframe =  read_df(self.data)
        self.assertEqual(dataframe.size, 387960, 'Wrong Size')
        self.assertEqual(dataframe.shape, (915, 424), 'Wrong Shape')
        self.assertIn('SITE', dataframe.keys(), 'SITE was not added')

    def test_add_WhiteVol_feature(self):
        dataframe = read_df(self.data)
        add_WhiteVol_feature(dataframe)
        self.assertIn('TotalWhiteVol', dataframe.keys(), 'TotalWhiteVol was not added')
        self.assertEqual(dataframe.shape, (915, 425), 'Wrong Shape')

    def test_df_split(self):
        dataframe = read_df(self.data)
        df_ASD, df_CTR = df_split(dataframe)
        assert df_ASD.shape == (451, 424)
        assert df_CTR.shape == (464, 424)

    def test_drop_covars(self):
        dataframe = read_df(self.data)
        dataframe, covar_list = drop_covars(dataframe)
        self.assertEqual(dataframe.shape, (915, 419), 'Wrong Shape')
        self.assertNotIn('SITE', dataframe.keys(), 'SITE was not dropped')
        self.assertNotIn('AGE_AT_SCAN', dataframe.keys(), 'AGE_AT_SCAN was not dropped')

    def test_normalization(self):
        dataframe = read_df(self.data)
        add_WhiteVol_feature(dataframe)
        normalization(dataframe)
        self.assertTrue(all(dataframe['TotalWhiteVol']<1),
                        "Dataframe was not normalized")



if __name__ == "__main__":
    unittest.main()
