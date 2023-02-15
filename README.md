[![Documentation Status](https://readthedocs.org/projects/brain-age-predictor/badge/?version=latest)](https://brain-age-predictor.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/pastiera/brain_age_predictor)](https://github.com/pastiera/brain_age_predictor/blob/main/LICENSE)
[![CircleCI](https://circleci.com/gh/Pastiera/brain_age_predictor.svg?style=svg)](https://circleci.com/gh/Pastiera/brain_age_predictor)
## brain_age_predictor

This repository contains a project for Computing Methods for Experimental Physics and Data Analysis course.

The aim is to design and implement a regression model to predict the age of the healthy subjects from brain data features extracted from T1-weighted MRI images. Datas are taken from to the well known [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) dataset, in which are present subjects affected by Autism Spectre Disorder (ASD) and healthy control subjects (CTR).

The algorithm allows to:
- visualize and explore ABIDE datas;
- make data harmonization by site;
- train different regression models using different cross validation;
- confront two alternative approaches to the problem.

The repository is structured as follows:
```
brain_age_predictor/
├── docs
├── LICENSE
├── dataset/
├── brain_age_predictor/
│   ├── images/
│   ├── metrics/
│   ├── best_estimator/
│   ├── preprocess.py
│   ├── brain_age_pred.py
│   ├── brain_age_site.py
│   ├── grid_CV.py
│   ├── loso_CV.py
│   ├── __init__.py
│   ├── variability.py
│   ├── DDNregressor.py
│   └── predict_helper.py
│   
├── README.md
├── requirements.txt
└── tests
    └── test.py
    └── __init__.py
```
# Data

Datas from ABIDE (Autism Brain Imaging Data Exchange) are contained in .csv files inside brain_age_predictor/dataset folder and are handled with Pandas. This dataset contains 419 brain morphological features (volumes, thickness, area, etc.) of different brain segmented area (via Freesurfer sofware) belonging to 915 male subjects (451 cases, 464 controls) pespectively with with total mean age of 
17.47 ± 0.36 and 17.38 ± 0.40. 
The age distribution of subjects, although heterogeneous between CTR and ASD groups, presents quite a skewed profile, as shown below:

<img src="brain_age_predictor/images/AGE_AT_SCAN_histogram.png" width="700"/>

Also age distribution across sites change quite drastically as shown in the following boxplot:

<img src="brain_age_predictor/images/AGE_distribution_box plot.png" width="700"/>

# Site harmonization

On top of these differencies, another important confounding factor is related to the effect of the different acquisition sites on the features. To mitigate this effect, the state-of-art harmonization tool [neuroHarmonize](https://github.com/rpomponio/neuroHarmonize) implemented by [Pomponio et al.](https://www.sciencedirect.com/science/article/pii/S1053811919310419?via%3Dihub) has been used.
<img src="brain_age_predictor/images/Unharmonized ABIDE dataframe_box plot.png" width="700"/>
<img src="brain_age_predictor/images/Harmonized ABIDE_box plot.png" width="700"/>

neuroHarmonize corrects differences introducted by multi-site image acquisition preserving specified covariates. So, harmonization can be safely performed without affecting age-related biological variability of the dataset.
This is particulary important as different sites have different age distribution.
The analysis has been conducted using 'unharmonized' and 'harmonized' datas.

# Analysis

## Method
Models have been trained using only control cases (CTR) and then evaluated separately on CTR set and cases set (ASD). Differences through residual plots are shown in the results avalaible in /images folder.
Being very poorly represented (<4%), subjects with age >40 years have been discarded from the present study, similarly to other studies in the field.[(1)](https://www.sciencedirect.com/science/article/pii/S2213158222001474)

## Pipelines
Two different pipelines have been followed based on Leave-One-Site-Out approach:
- **1**)  Datas have been previously separeted in train/test sets using one provenance site as test and the others as train and consequently cross-validated with KFold CV.[(2)](https://pubmed.ncbi.nlm.nih.gov/34924987/)[(3)](https://link.springer.com/content/pdf/10.1007/s12021-018-9366-0.pdf)
- **2**)  Datas have been processed without discrimination based on site, but using different cross validation approaches: one using a custom Leave-One-Site-Out(LOSO) cross validation, the other a regular [GridSearch CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
Both scikitlearn's models and a custom neural network have been used.
# Results
Typical regression metrics (MAE, MSE) have been evaluated. Pearson correlation coefficient (PR) has been also calculated too. For pipeline 1, results' plots are collected in 'images' folder, while fitted models and relative metrics' results are stored respectively in 'best_estimator' and 'metrics/grid(loso)' folders. For pipeline 2, results' metrics are also stored in 'metrics/site' and summarizing plots are stored in 'images_SITE' folder.


<img src="brain_age_predictor/images/losoCV/Sites NeuroHarmonized with Random_Forest_Regressor.png" width="700"/> 
<img src="brain_age_predictor/images/losoCV/df_CTR_test_Harmonized_Random_Forest_Regressor.png" width="700"/>


# Requirements
To use these Python codes the following packages are required: 
- keras
- matplotlib
- neuroHarmonize
- numpy
- pandas
- prettytable
- scikit-learn
- scipy
- seaborn
- sphinx
- statsmodels
- tensorflow

# Usage

- **1**) Download the repository from github
```git clone https://github.com/Pastiera/brain_age_predictor```
- **2**) Change directory: ```cd path/to/brain_age_predictor/brain_age_predictor```
- **3**) Modules brain_age_pred.py, brain_age_site.py, variability.py, preprocess.py are executable following relative help instruction by typing -h on std-out line as positional argument.
Example:
```
brain_age_pred.py [-h]

optional arguments:
  -h, --help            show this help message and exit
  -dp DATAPATH, --datapath DATAPATH
                        Path to the data folder.
  -loso, --losocv       Use Leave-One-Site-Out CV to train and fit models.
  -grid, --gridcv       Use GridSearch cross validation to train and fit models.
  -fitgrid, --fitgridcv
                        Make predictions with models pre-trainde with GridSearchCV.
  -fitloso, --fitlosocv
                        Make predictions with models pre-trainde with LOSO-CV.
  -neuroharm, --harmonize
                        Use NeuroHarmonize to harmonize data by provenance site.

```

Pre-trained model in /best_estimator can be run for reproducibility and newly trained model will be saved in the same folder. If no fitted models is already present in this folder, one shall firstly run ```brain_age_pred.py``` to use ```variability.py```.
Results' plots are collected in 'images' or 'images_site' folder, while fitted models and relative metrics' results are stored respectively in 'best_estimator' and 'metrics' folders.

## References

- [1] [Courchesne E, Campbell K, Solso S. Brain growth across the life span in autism: age-specific changes in anatomical pathology. Brain Res. 2011 Mar 22;1380:138-45. doi: 10.1016/j.brainres.2010.09.101. Epub 2010 Oct 1. PMID: 20920490; PMCID: PMC4500507.](https://pubmed.ncbi.nlm.nih.gov/20920490/)
- [2] [Saponaro S., Giuliano A., Bellotti R.,Lombardi A., Tangaro S.,Oliva P., Calderoni S., Retico A., Multi-site harmonization of MRI data uncovers machine-learning discrimination capability in barely separable populations: An example from the ABIDE dataset, NeuroImage: Clinical, Volume 35, 2022, 103082, ISSN 2213-1582](https://www.sciencedirect.com/science/article/pii/S2213158222001474)
- [3] [Okamoto N and Akama H (2021), Extended Invariant Information Clustering Is Effective for Leave-One-Site-Out Cross-Validation in Resting State Functional Connectivity Modeling. Front.Neuroinform. 15:709179.](https://pubmed.ncbi.nlm.nih.gov/34924987/)
- [4] [Bhaumik, R., Pradhan, A., Das, S. et al. Predicting Autism Spectrum Disorder Using Domain-Adaptive Cross-Site Evaluation. Neuroinform 16, 197–205 (2018).](https://link.springer.com/content/pdf/10.1007/s12021-018-9366-0.pdf)

