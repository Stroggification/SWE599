# SWE599
Repository for final term project SWE 599

This repository includes scripts for synchronizing and merging the ECSMP and SWEET data sets. Explanations are given in the report and comments.

Here is the order of scripts and explanations 

ECSMPresampler.py

resamples the ecsmp dataset to a common sampling rate and merges the files into a single file per user

SWEETresampler.py

resamples the sweet dataset to a common sampling rate and merges the files into a single file per user

missing_feature_detector.py

checks the resampled and merged data for any missing features(columns)

data_analysis.py

diagnosis and visualizes the data for validation and integrity check

outlierfix.py

checks for any outliear data. in our case only outlier data was in sweet dataset temparature valeus of few users so you dont have to run this sciprt for ECSMP dataset

data_validation.py

fixes the data imbalances(overrepresented classes)  and missign values in the datasets(by imputing) and saves the new balanced dataset.



