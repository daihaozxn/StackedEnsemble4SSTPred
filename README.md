# StackedEnsemble4SSTPred
This is a repository for the stacked ensemble method to improve predicting SST in Taiwan Strait and East China Sea.

In order to replicate the promotion effect of the stacked ensemble method in the paper quickly, first load the saved 0th-level model in the StackedwithConvLSTMMetaModel.py, and then train and test the 1st-level model.

The stacked ensemble method is implemented with the deep learning framework Keras with Tensorflow 2.3.0 (GPU version, Python 3.7) as the backend and the Integrated Development Environment Pycharm.

The folder "data" (unzipped by "data.zip") is used to store source data, where raw_train.h5 and raw_test.h5 are source data for training and test of Taiwan Strait while donghai_raw_train.h5 and donghai_raw_test.h5 are source data for training and test of East China Sea. They are sourced from NOAA/OAR/ESRL PSL at the website https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html.

The folder "Submodels" (unzipped by "Submodels.zip") is used to store the saved trained 0th-level models, where folders "Lead_time_1/3/5/7" are used to store the saved trained 0th-level models for lead time 1/3/5/7 days of Taiwan Strait, and folders "DH_lead_time_1/3/5/7" are used to store the saved trained 0th-level models for lead time 1/3/5/7 days of East China Sea.
