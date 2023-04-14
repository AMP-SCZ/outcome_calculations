import sys
import numpy as np
import pandas as pd
import argparse
import json
import os
from datetime import datetime
from datetime import date

# --------------------------------------------------------------------#
# In this script we calculate all the outcomes for the AMP-SCZ study.
# --------------------------------------------------------------------#

# --------------------------------------------------------------------#
# We first have to define all functions needed for the outcome 
# calculations later. 
# --------------------------------------------------------------------#

# Here, we load the data that was created on March, 26th and check them!
test_subj_pronet = pd.read_csv("/data/predict1/home/np487/amp_scz/pronet_test_subjects.csv")
test_subj_prescient = pd.read_csv("/data/predict1/home/np487/amp_scz/prescient_test_subjects.csv")
test_subj = pd.concat([test_subj_pronet, test_subj_prescient], axis = 0)

new_subj_pronet = pd.read_csv("/data/predict1/home/np487/amp_scz/test_subjects/pronet_new_subjects.csv")
new_subj_prescient = pd.read_csv("/data/predict1/home/np487/amp_scz/test_subjects/prescient_new_subjects.csv")
new_subj = pd.concat([new_subj_pronet, new_subj_prescient], axis = 0)

test_subj['value_original']=test_subj['value']
new_subj['value_totest_new']=new_subj['value']
test_subj['type_original']=test_subj['data_type']
new_subj['type_totest_new']=new_subj['data_type']


df_to_test = pd.merge(test_subj, new_subj, on = ['ID', 'variable', 'redcap_event_name'], how = 'outer')
df_to_test['value_totest_new']=df_to_test['value_totest_new'].astype(str)
df_to_test['value_original']=df_to_test['value_original'].astype(str) 

# test whether the dataframes are different
filtered_df = df_to_test[df_to_test['value_original'] != df_to_test['value_totest_new']]
filtered_df = filtered_df[['ID', 'variable', 'redcap_event_name', 'type_original','type_totest_new', 'value_original', 'value_totest_new']]
num_rows, num_cols = filtered_df.shape


if num_rows > 0:
    print("filtered_df")
    print(filtered_df.to_string())
    print("There seems to be a problem with the code and before running it we have to check this!")
    print("Number of rows:", num_rows)
elif num_rows == 0:
    print("The data is the same as when we have checked it. Thus, go ahead and run the code")
    print("Number of rows:", num_rows)






