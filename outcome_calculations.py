import sys
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import argparse
import json
import os
from datetime import datetime, date
import time
import multiprocessing

# --------------------------------------------------------------------#
# In this script we calculate all the outcomes for the AMP-SCZ study.
# --------------------------------------------------------------------#

# --------------------------------------------------------------------#
# We first have to define all functions needed for the outcome 
# calculations later. 
# --------------------------------------------------------------------#
def pull_data(network, id):
    # we pull the data from the json file. Therefore, we provide the network, i.e. pronet/prescient.
    # network = pronet/prescient defined outside the script argv[0], id = subject id, taken from the updated file that Grace creates daily.
    site=id[0:2]
    sub_data = '/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/{2}.{0}.json'.format(network, site, id)
    with open(sub_data, 'r') as f:
            json_data = json.load(f)
    sub_data_all = pd.DataFrame.from_dict(json_data, orient="columns")
    #replacing empty cells with NaN
    sub_data_all = sub_data_all.apply(lambda x: x.str.strip()).replace('', np.nan)
    return sub_data_all

def create_fake_df(var_list, all_visits, voi):
    # we create a fake dataframe that includes all visits and the outcome measures.
    # var_list = list with all variables that are needed for the calculation. 
    # all_visits = list with all visist of the amp-scz, 
    # voi = visits that are applicable for this outcome.
    df_fake = {'variable': [var_list]*len(all_visits), 'redcap_event_name': all_visits}
    df_fake = pd.DataFrame(df_fake)  
    # by default we give all the values a -300 if not changed afterwards
    df_fake['value_fake'] = '-300'
    # vois = visits of interest. We change the values from -300 (not applicable) to -900 (missing) for all the visits 
    # for which we actually expect data.
    df_fake['value_fake'] = np.where(df_fake['redcap_event_name'].str.contains(voi), '-900', df_fake['value_fake'])
    return df_fake 

def create_fake_df_float(var_list, all_visits, voi):
    # we create a fake dataframe that includes all visits and the outcome measures.
    # var_list = list with all variables that are needed for the calculation. 
    # all_visits = list with all visist of the amp-scz, 
    # voi = visits that are applicable for this outcome.
    df_fake = {'variable': [var_list]*len(all_visits), 'redcap_event_name': all_visits}
    df_fake = pd.DataFrame(df_fake)  
    # by default we give all the values a -300 if not changed afterwards
    df_fake['value_fake'] = '-300.000'
    # vois = visits of interest. We change the values from -300 (not applicable) to -900 (missing) for all the visits 
    # for which we actually expect data.
    df_fake['value_fake'] = np.where(df_fake['redcap_event_name'].str.contains(voi), '-900.000', df_fake['value_fake'])
    df_fake['value'] = np.round(df_fake['value_fake'].astype(float),1)
    df_fake = df_fake[['variable', 'redcap_event_name', 'value']]
    df_fake['variable'] = df_fake['variable'].astype(str)
    df_fake['variable'] = df_fake['variable'].str.strip("['']")
    return df_fake 

def create_fake_df_date(var_list, all_visits, voi):
    # we create a fake dataframe that includes all visits and the outcome measures.
    # var_list = list with all variables that are needed for the calculation. 
    # all_visits = list with all visist of the amp-scz, 
    # voi = visits that are applicable for this outcome.
    df_fake = {'variable': [var_list]*len(all_visits), 'redcap_event_name': all_visits}
    df_fake = pd.DataFrame(df_fake)  
    df_fake['value_fake'] = '1903-03-03'
    df_fake['value_fake'] = np.where(df_fake['redcap_event_name'].str.contains(voi), '1909-09-09', df_fake['value_fake'])
    return df_fake 

def finalize_df(df_created, df_1, df_2, var_list, voi, fill_type):
    # used for all dataframes in the end to create it in the way we need it for appropriate handling of all timepoints etc.(creating a clean version)
    # df_created = comes from the create_fake_df function. A dataframe that includes all variables, visits and initially value_fake with -300
    # df_1 = the dataframe with the actual calculated value 
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # voi = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # fill_type = defines the final output variable. Either float or int
    string_series = pd.Series(var_list)
    df_visit_pas = df_2[['redcap_event_name']]
    # Here, we account for missingness (-900) and not applicableness (-300) and change values 
    # to the appropriate missingcodes (e.g., 999 -> -900)
    if all(col in df_2.columns for col in var_list):
        df_2_vars = df_2.copy()
        df_1_vars = df_1.copy()
        df_2_vars = df_2_vars[['redcap_event_name'] + var_list]
        df_1_vars = df_1.copy()
        df_1_vars = df_1_vars[['redcap_event_name'] + var_list]
        merged_df = pd.merge(df_1_vars, df_2_vars, on = 'redcap_event_name', how = 'left', suffixes = ('_left', ''))
        merged_df[var_list] = merged_df[var_list].fillna(333)
        var_list_plus = var_list + [f'{col}_left' for col in var_list]
        df_1 = df_1.reset_index(drop=True)
        merged_df = merged_df.reset_index(drop=True)
    if string_series.str.contains('nsipr').any() == True or string_series.str.contains('chrpas').any() == True:
        df_1['nine'] = df_1[var_list].isin([9]).any(axis=1)
        df_1['nine_str'] = df_1[var_list].isin(['9']).any(axis=1)
        if all(col in df_2.columns for col in var_list):
            df_1['nine'] = merged_df[var_list_plus].isin([9]).any(axis=1)
            df_1['nine_str'] = merged_df[var_list_plus].isin(['9']).any(axis=1)
    # Check if the columns in var_list are present in df_2 before applying the logic
    if all(col in df_2.columns for col in var_list):
        df_1['NA2'] = merged_df[var_list_plus].isin(['N/A']).any(axis=1)            #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-9']).any(axis=1)            
        df_1['NULL'] = merged_df[var_list_plus].isin(['NULL']).any(axis=1)            #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-9']).any(axis=1)            
        df_1['empty'] = merged_df[var_list_plus].isin(['']).any(axis=1)            #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-9']).any(axis=1)            
        df_1['na_NaN'] = merged_df[var_list_plus].isna().any(axis=1)            #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-9']).any(axis=1)            
        df_1['na9'] = merged_df[var_list_plus].isin(['-9']).any(axis=1)            #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-9']).any(axis=1)            
        df_1['na']  = merged_df[var_list_plus].isnull().any(axis = 1)              #| merged_df[[f'{col}_left' for col in var_list_plus]].isnull.any(axis=1)                       
        df_1['na2'] = merged_df[var_list_plus].isin(['n/a']).any(axis = 1)         #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['n/a']).any(axis=1)            
        df_1['na22'] = merged_df[var_list_plus].isin(['na']).any(axis = 1)         #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['na']).any(axis=1)            
        df_1['NaN'] = merged_df[var_list_plus].isin(['NaN']).any(axis = 1)         #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['NaN']).any(axis=1)             
        df_1['NA'] = merged_df[var_list_plus].isin(['NA']).any(axis = 1)           #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['NA']).any(axis=1)            
        df_1['na999'] = merged_df[var_list_plus].isin(['999']).any(axis = 1)       #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['999']).any(axis=1)            
        df_1['na999nostring'] = merged_df[var_list_plus].isin([999]).any(axis = 1) #| merged_df[[f'{col}_left' for col in var_list_plus]].isin([999]).any(axis=1)            
        df_1['na900'] = merged_df[var_list_plus].isin(['-900']).any(axis = 1)      #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-900']).any(axis=1)            
        df_1['na900nostring'] = merged_df[var_list_plus].isin([-900]).any(axis = 1)#| merged_df[[f'{col}_left' for col in var_list_plus]].isin([-900]).any(axis=1)            
        df_1['na300'] = merged_df[var_list_plus].isin(['-300']).any(axis = 1)      #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-300']).any(axis=1)            
        df_1['na300nostring'] = merged_df[var_list_plus].isin([-300]).any(axis = 1)#| merged_df[[f'{col}_left' for col in var_list_plus]].isin([-300]).any(axis=1)            
        df_1['na3'] = merged_df[var_list_plus].isin(['-3']).any(axis = 1)          #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-3']).any(axis=1)            
        df_1['na3nostring'] = merged_df[var_list_plus].isin([-3]).any(axis = 1)    #| merged_df[[f'{col}_left' for col in var_list_plus]].isin([-3]).any(axis=1)            
        df_1['na99'] = merged_df[var_list_plus].isin(['-99']).any(axis = 1)        #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-99']).any(axis=1)            
        df_1['na99nostring'] = merged_df[var_list_plus].isin([-99]).any(axis = 1)  #| merged_df[[f'{col}_left' for col in var_list_plus]].isin([-99]).any(axis=1)            
        df_1['na9'] = merged_df[var_list_plus].isin(['-9']).any(axis = 1)          #| merged_df[[f'{col}_left' for col in var_list_plus]].isin(['-9']).any(axis=1)            
        df_1['na9nostring'] = merged_df[var_list_plus].isin([-9]).any(axis = 1)    #| merged_df[[f'{col}_left' for col in var_list_plus]].isin([-9]).any(axis=1)            
    else:
        df_1['NA2'] = df_1[var_list].isin(['N/A']).any(axis = 1)
        df_1['NULL'] = df_1[var_list].isin(['NULL']).any(axis = 1)
        df_1['empty'] = df_1[var_list].isin(['empty']).any(axis = 1)
        df_1['na_NaN'] = df_1[var_list].isna().any(axis = 1)
        df_1['na'] = df_1[var_list].isnull().any(axis = 1)
        df_1['na2'] = df_1[var_list].isin(['n/a']).any(axis = 1)
        df_1['na22'] = df_1[var_list].isin(['na']).any(axis = 1)
        df_1['NaN'] = df_1[var_list].isin(['NaN']).any(axis = 1)
        df_1['NA'] = df_1[var_list].isin(['NA']).any(axis = 1)
        df_1['na999'] = df_1[var_list].isin(['999']).any(axis = 1)
        df_1['na999nostring'] = df_1[var_list].isin([999]).any(axis = 1)
        df_1['na900'] = df_1[var_list].isin(['-900']).any(axis = 1)
        df_1['na900nostring'] = df_1[var_list].isin([-900]).any(axis = 1)
        df_1['na300'] = df_1[var_list].isin(['-300']).any(axis = 1)
        df_1['na300nostring'] = df_1[var_list].isin([-300]).any(axis = 1)
        df_1['na3'] = df_1[var_list].isin(['-3']).any(axis = 1)
        df_1['na3nostring'] = df_1[var_list].isin([-3]).any(axis = 1)
        df_1['na99'] = df_1[var_list].isin(['-99']).any(axis = 1)
        df_1['na99nostring'] = df_1[var_list].isin([-99]).any(axis = 1)
        df_1['na9'] = df_1[var_list].isin(['-9']).any(axis = 1)
        df_1['na9nostring'] = df_1[var_list].isin([-9]).any(axis = 1)
    if string_series.str.contains('nsipr').any() == True or string_series.str.contains('chrpas').any() == True:
        df_calculated = df_1[['value','nine','nine_str', 'na', 'empty', 'na_NaN', 'na22','na2', 'NA', 'NaN', 'na999nostring', 'na999', 'na900nostring','na900', 'na300nostring', 'na300', 'na3nostring','na3', 'na99', 'na99nostring',\
                              'na9nostring','na9', 'redcap_event_name']]
    else:
        df_calculated = df_1[['value', 'NULL', 'NA2', 'na', 'empty', 'na_NaN', 'na2', 'na22','NA', 'NaN', 'na999nostring', 'na999', 'na900nostring','na900', 'na300nostring', 'na300', 'na3nostring','na3','na99', 'na99nostring', 'na9nostring','na9', 'redcap_event_name']]
    df_2['redcap_event_name'] = df_2['redcap_event_name'].astype(str)
    df_visits = df_2[['redcap_event_name']]
    df_visits = df_visits[df_visits['redcap_event_name'].str.contains(voi)]
    df_concat = pd.merge(df_visits, df_calculated, on = 'redcap_event_name', how = 'left')
    df2gether = pd.merge(df_created, df_concat, on = 'redcap_event_name', how = 'left')
    if fill_type == 'str':
        df2gether['value'] = np.where(df2gether['na2'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na22'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na999'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na999nostring'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na900'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na900nostring'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na300'] == True, '-300', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na300nostring'] == True, '-300', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na3'] == True, '-300', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na3nostring'] == True, '-300', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na99'] == True, '-300', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na99nostring'] == True, '-300', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na9'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na9nostring'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(df2gether['NA'] == True, '-900', df2gether['value'])
        df2gether['value'] = np.where(pd.isna(df2gether['value']),df2gether['value_fake'], df2gether['value'])
        if df_visits['redcap_event_name'].str.contains('arm_1').any():
            df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_2'), '-300', df2gether['value'])
        elif df_visits['redcap_event_name'].str.contains('arm_2').any():
            df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_1'), '-300', df2gether['value'])
        elif df_visit_pas['redcap_event_name'].str.contains('arm_2').any() and voi == 'month_1_arm_1':
            df2gether['value'] = -300
        df2gether['value'] = df2gether['value'].astype(str)
        df2gether['value'] = np.where(pd.isna(df2gether['value']),df2gether['value_fake'], df2gether['value'])
    else:
        if string_series.str.contains('nsipr').any() == True or string_series.str.contains('chrpas').any() == True:
            df2gether['value'] = np.where(df2gether['nine'] == True, -900, df2gether['value'])
            df2gether['value'] = np.where(df2gether['nine_str'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na2'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na22'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na999'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na999nostring'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na900'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na900nostring'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na300'] == True, -300, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na300nostring'] == True, -300, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na3'] == True, -300, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na3nostring'] == True, -300, df2gether['value'])
        # keep in mind that the -99 might be changed to a missing instead of a non-applicable
        df2gether['value'] = np.where(df2gether['na99'] == True, -300, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na99nostring'] == True, -300, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na9'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['na9nostring'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(df2gether['NA'] == True, -900, df2gether['value'])
        df2gether['value'] = np.where(pd.isna(df2gether['value']),df2gether['value_fake'], df2gether['value'])
        if df_visits['redcap_event_name'].str.contains('arm_1').any():
            df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_2'), -300, df2gether['value'])
        elif df_visits['redcap_event_name'].str.contains('arm_2').any():
            df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_1'), -300, df2gether['value'])
        elif df_visit_pas['redcap_event_name'].str.contains('arm_2').any() and voi == 'month_1_arm_1':
            df2gether['value'] = -300
        df2gether['value'] = df2gether['value'].astype(float)
        df2gether['value'] = np.where(pd.isna(df2gether['value']),df2gether['value_fake'], df2gether['value'])
    clean_df = df2gether[['variable', 'redcap_event_name', 'value']]
    if fill_type == 'float':
        clean_df.loc[:,'value'] = np.round(clean_df['value'].astype(fill_type),3)
    else:
        clean_df.loc[:,'value'] = clean_df['value'].astype(fill_type)
    return clean_df

def finalize_df_scid(df_created, df_1, df_2, var_list, voi, fill_type):
    # used for all dataframes in the end to create it in the way we need it for appropriate handling of all timepoints etc.(creating a clean version)
    # df_created = comes from the create_fake_df function. A dataframe that includes all variables, visits and initially value_fake with -300
    # df_1 = the dataframe with the actual calculated value 
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # voi = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # fill_type = defines the final output variable. Either float or int
    string_series = pd.Series(var_list)
    df_visit_pas = df_2[['redcap_event_name']]
    df_calculated = df_1[['value', 'redcap_event_name']]
    df_2['redcap_event_name'] = df_2['redcap_event_name'].astype(str)
    df_visits = df_2[['redcap_event_name']]
    df_visits = df_visits[df_visits['redcap_event_name'].str.contains(voi)]
    df_concat = pd.merge(df_visits, df_calculated, on = 'redcap_event_name', how = 'left')
    df2gether = pd.merge(df_created, df_concat, on = 'redcap_event_name', how = 'left')
    if df_visits['redcap_event_name'].str.contains('arm_1').any():
        df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_2'), -300, df2gether['value'])
    elif df_visits['redcap_event_name'].str.contains('arm_2').any():
        df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_1'), -300, df2gether['value'])
    elif df_visit_pas['redcap_event_name'].str.contains('arm_2').any() and voi == 'month_1_arm_1':
        df2gether['value'] = -300
    df2gether['value'] = df2gether['value'].astype(float)
    df2gether['value'] = np.where(pd.isna(df2gether['value']),df2gether['value_fake'], df2gether['value'])
    clean_df = df2gether[['variable', 'redcap_event_name', 'value']]
    if fill_type == 'float':
        clean_df.loc[:,'value'] = np.round(clean_df['value'].astype(fill_type),3)
    else:
        clean_df.loc[:,'value'] = clean_df['value'].astype(fill_type)
    return clean_df

def finalize_df_date(df_created, df_1, df_2, var_list, voi, fill_type):
    # used for all dataframes in the end to create it in the way we need it for appropriate handling of all timepoints etc.(creating a clean version)
    # df_created = comes from the create_fake_df function. A dataframe that includes all variables, visits and initially value_fake with -300
    # df_1 = the dataframe with the actual calculated value 
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # voi = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # fill_type = defines the final output variable. Either float or int
    string_series = pd.Series(var_list)
    df_visit_pas = df_2[['redcap_event_name']]
    # Here, we account for missingness (-900) and not applicableness (-300) and change values 
    # to the appropriate missingcodes (e.g., 999 -> -900)
    df_1['na'] = df_1[var_list].isnull().any(axis = 1)
    df_1['NA'] = df_1[var_list].isin(['NA']).any(axis = 1)
    df_1['na999'] = df_1[var_list].isin(['999']).any(axis = 1)
    df_1['na999nostring'] = df_1[var_list].isin([999]).any(axis = 1)
    df_1['na900'] = df_1[var_list].isin(['-900']).any(axis = 1)
    df_1['na900nostring'] = df_1[var_list].isin([-900]).any(axis = 1)
    df_1['na300'] = df_1[var_list].isin(['-300']).any(axis = 1)
    df_1['na300nostring'] = df_1[var_list].isin([-300]).any(axis = 1)
    df_1['na3'] = df_1[var_list].isin(['-3']).any(axis = 1)
    df_1['na3nostring'] = df_1[var_list].isin([-3]).any(axis = 1)
    df_1['na9'] = df_1[var_list].isin(['-9']).any(axis = 1)
    df_1['na9nostring'] = df_1[var_list].isin([-9]).any(axis = 1)
    df_1['nadate9'] = df_1[var_list].isin(['1909-09-09']).any(axis = 1)
    df_1['nadate3'] = df_1[var_list].isin(['1903-03-03']).any(axis = 1)
    df_calculated = df_1[['value', 'nadate3', 'nadate9','na', 'NA', 'na999nostring', 'na999', 'na900nostring','na900', 'na300nostring', 'na300', 'na3nostring','na3', 'na9nostring','na9', 'redcap_event_name']]
    df_2['redcap_event_name'] = df_2['redcap_event_name'].astype(str)
    df_visits = df_2[['redcap_event_name']]
    df_visits = df_visits[df_visits['redcap_event_name'].str.contains(voi)]
    df_concat = pd.merge(df_visits, df_calculated, on = 'redcap_event_name', how = 'left')
    df2gether = pd.merge(df_created, df_concat, on = 'redcap_event_name', how = 'left')
    if fill_type == 'str':
        df2gether['value'] = np.where(df2gether['nadate9'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['nadate3'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na999'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na999nostring'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na900'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na900nostring'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na300'] == True, '1903-03-03', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na300nostring'] == True, '1903-03-03', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na3'] == True, '1903-03-03', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na3nostring'] == True, '1903-03-03', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na9'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['na9nostring'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(df2gether['NA'] == True, '1909-09-09', df2gether['value'])
        df2gether['value'] = np.where(pd.isna(df2gether['value']),df2gether['value_fake'], df2gether['value'])
        if df_visits['redcap_event_name'].str.contains('arm_1').any():
            df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_2'), '1903-03-03', df2gether['value'])
        elif df_visits['redcap_event_name'].str.contains('arm_2').any():
            df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_1'), '1903-03-03', df2gether['value'])
        elif df_visit_pas['redcap_event_name'].str.contains('arm_2').any() and voi == 'month_1_arm_1':
            df2gether['value'] = '1903-03-03'
        df2gether['value'] = df2gether['value'].astype(str)
        df2gether['value'] = np.where(pd.isna(df2gether['value']),df2gether['value_fake'], df2gether['value'])
    clean_df = df2gether[['variable', 'redcap_event_name', 'value']]
    clean_df.loc[:,'value'] = clean_df['value'].astype(fill_type)
    return clean_df

def create_total_division(outcome, df_1, df_2, var_list, division, visit_of_interest, all_visits, fill_type):
    # we create the total (sum) score and if applicable divide it.
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    df_fake = create_fake_df(outcome, all_visits, visit_of_interest)
    if df_2['redcap_event_name'].str.contains('arm_1').any():
        df_1 = df_1[df_1['redcap_event_name'].str.contains('arm_1')]
    elif df_2['redcap_event_name'].str.contains('arm_2').any():
        df_1 = df_1[df_1['redcap_event_name'].str.contains('arm_2')]
    df_1 = df_1[df_1['redcap_event_name'].str.contains(visit_of_interest)]
    df_1['value'] = df_1[var_list].fillna(-900).astype(fill_type).sum(axis = 1)/division
    final_df = finalize_df(df_fake, df_1, df_2, var_list, visit_of_interest, fill_type)
    return final_df 

def create_max(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type):
    # we create the total (max) score 
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    df_fake = create_fake_df(outcome, all_visits, visit_of_interest)
    df_1[var_list] = df_1[var_list].replace('NaN', np.nan)
    df_1['value'] = df_1[var_list].fillna(-900).astype(fill_type).max(axis = 1)
    final_df = finalize_df(df_fake, df_1, df_2, var_list, visit_of_interest, fill_type)
    return final_df 

def create_max_scid(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type):
    # we create the total (max) score 
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    df_fake = create_fake_df(outcome, all_visits, visit_of_interest)
    df_1[var_list] = df_1[var_list].replace('NaN', np.nan)
    df_1['value'] = df_1[var_list].fillna(-900).astype(fill_type).max(axis = 1)
    final_df = finalize_df_scid(df_fake, df_1, df_2, var_list, visit_of_interest, fill_type)
    return final_df 

def create_min_date(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type):
    # we create the total (min) date
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    df_fake = create_fake_df_date(outcome, all_visits, visit_of_interest)
    pd.set_option('display.max_columns', None)
    # the problem here is that I have to update again the date calculation because at the moment it always will pick the dates with 1909-09-09
    df_1[var_list] = df_1[var_list].fillna('2090-09-09').astype(fill_type)
    df_1[var_list] = np.where(df_1[var_list] == '1909-09-09', '2090-09-09', df_1[var_list])
    df_1[var_list] = np.where(df_1[var_list] == '1903-03-03', '2033-03-03', df_1[var_list])
    df_1[var_list] = df_1[var_list].apply(pd.to_datetime, format = '%Y-%m-%d', errors = 'coerce')
    df_1['value'] = df_1[var_list].min(axis = 1)
    df_1['value'] = df_1['value'].astype(fill_type)
    df_1['value'] = np.where(df_1['value'] == '2090-09-09', '1909-09-09', df_1['value'])
    df_1['value'] = np.where(df_1['value'] == '2033-03-03', '1903-03-03', df_1['value'])
    df_1[var_list] = df_1[var_list].astype(fill_type) 
    final_df = finalize_df_date(df_fake, df_1, df_2, var_list, visit_of_interest, fill_type)
    final_df['value'] = pd.to_datetime(final_df['value'], format='%Y-%m-%d').dt.date
    return final_df 

def create_mul(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type):
    # we create the total (multiplied) score.
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    df_fake = create_fake_df(outcome, all_visits, visit_of_interest)
    df_1['value'] = df_1[var_list].fillna(-900).astype(fill_type).prod(axis = 1)
    final_df = finalize_df(df_fake, df_1, df_2, var_list, visit_of_interest, fill_type)
    return final_df 

def create_decline(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type):
    # we create the subtraction between two variables
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    df_fake = create_fake_df(outcome, all_visits, visit_of_interest)
    df_1[var_list[0]] = df_1[var_list[0]].fillna(-900).astype(fill_type)
    df_1[var_list[1]] = df_1[var_list[1]].fillna(-900).astype(fill_type)
    df_1['value'] = df_1[var_list[0]] - df_1[var_list[1]]
    final_df = finalize_df(df_fake, df_1, df_2, var_list, visit_of_interest, fill_type)
    return final_df 

def create_use_value(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type):
    # we create the total (sum) score and if applicable divide it.
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # var_list = list of all variables needed for the calculation
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    df_fake = create_fake_df(outcome, all_visits, visit_of_interest)
    df_1['value'] = df_1[var_list[0]]
    final_df = finalize_df(df_fake, df_1, df_2, var_list, visit_of_interest, fill_type)
    return final_df 

def create_condition_value(outcome, df_1, df_2, visit_of_interest, all_visits, fill_type, given_value):
    # after setting up the specific condition (if), we create the dataframe with the given value. 
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    # given_value = as soon as a condition is met we have defined values from Dominic Oliver that are filled in here.
    df_fake = create_fake_df(outcome, all_visits, visit_of_interest)
    df_1['value'] = given_value
    df_calculated = df_1['value']
    df_2['redcap_event_name'] = df_2['redcap_event_name'].astype(str)
    df_visits = df_2[['redcap_event_name']]
    df_visits = df_visits[df_visits['redcap_event_name'].str.contains(visit_of_interest)]
    df_concat = pd.concat([df_visits, df_calculated], axis = 1)
    df2gether = pd.merge(df_fake, df_concat, on='redcap_event_name', how='left')
    df2gether = df2gether.fillna(np.nan).astype('object')
    #df2gether = pd.merge(df_fake, df_concat, on = 'redcap_event_name', how = 'left').replace(np.nan, pd.NA).astype('object')
    if df_visits['redcap_event_name'].str.contains('arm_1').any():
        df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_2'), '-300', df2gether['value'])
    elif df_visits['redcap_event_name'].str.contains('arm_2').any():
        df2gether['value'] = np.where(df2gether['redcap_event_name'].str.contains('arm_1'), '-300', df2gether['value'])
    df2gether['value'] = np.where(pd.isnull(df2gether['value']),df2gether['value_fake'], df2gether['value'])
    clean_df = df2gether[['variable', 'redcap_event_name', 'value']]
    clean_df.loc[:,'value'] = clean_df['value'].astype(fill_type)
    return clean_df 

def create_assist(outcome, df_1, df_2, assist_var1, assist_var2, var_list, division, visit_of_interest, all_visits, fill_type):
    # ASSIST is a special case because we need one extra variable (whether or not individuals use the substance itself. If not we do not need to calculate the sum.
    # outcome = string that defines the outcome name, e.g., bprs_total
    # df_1 = the dataframe including the actual variables needed to calculate the outcome. This is most of the time df_all, but sometimes is the dataframe created by previous calculations, e.g., promis_df
    # df_2 = the dataframe with all visits that are relevant for the participants. used to define whether individual is chr or hc
    # assist_var = variable whether individuals use the substance at all.
    # var_list = list of all variables needed for the calculation
    # division = as the common function (create_total_division) is called here we also give a division number (in this case 1). 
    # visit_of_interest = visit of interest for the variable of calculation to define whether missing values are -900 or -300
    # all_visits = list with all visist of the amp-scz, 
    # fill_type = defines the final output variable. Either float or int
    final_df = create_total_division(outcome, df_1, df_2, var_list, division, visit_of_interest, all_visits, fill_type)
    final_df_pastmonth_zero = create_total_division(outcome, df_1, df_2, var_list[4:5], division, visit_of_interest, all_visits, fill_type)
    final_df_pastmonth_zero['outcome_pastmonth_zero'] = final_df_pastmonth_zero['value']
    final_df_pastmonth_zero = final_df_pastmonth_zero[['outcome_pastmonth_zero', 'redcap_event_name']]
    df_1[assist_var1] = df_1[assist_var1].fillna(-900).astype(int)
    df_1['no_use'] = df_1[[assist_var1]].isin([0]).any(axis = 1)
    df_1[assist_var2] = df_1[assist_var2].fillna(-900).astype(int)
    df_1['no_use_pastmonth'] = df_1[[assist_var2]].isin([0]).any(axis = 1)
    df_1 = df_1[['no_use', 'no_use_pastmonth', 'redcap_event_name']]
    df_assist_created = pd.merge(final_df, df_1, on = 'redcap_event_name', how = 'left')
    df_assist_created = pd.merge(df_assist_created, final_df_pastmonth_zero, on = 'redcap_event_name', how = 'left')
    df_assist_created['value'] = np.where(df_assist_created['no_use'] == True, 0, df_assist_created['value'])
    df_assist_created['value'] = np.where(df_assist_created['no_use_pastmonth'] == True, df_assist_created['outcome_pastmonth_zero'], df_assist_created['value'])
    final_assist = df_assist_created[['variable', 'redcap_event_name', 'value']]
    return final_assist

def create_sips_groups_scr(new_sips_group, df, onsetdate_sips_group, visit_of_interest, all_visits, psychosis_onset_df, sips_vars_interest_screening,\
                           psychosis_str):
    # create the different sips group assignments at screening
    # df: dataframe (df_all) including all variables
    # onsetdate_sips_group: onset date of the sips-group for all symptoms
    # visit_of_interest: the visits (here probably voi_8) that are of interest for the questionnaire (psychs)
    # all_visits: list with all visits (all_visits_list)
    # psychosis_onset_df: dataframe with the psychosis conversion date (conversion_date_fu)
    # sips_vars_interest_fu: vars of interest for the follow-up visits (e.g. bips symptoms)
    # psychosis_str = variable indicating psychosis conversion or not *psychs_fu_ac1_conv
    # new_sips_group = string iwth the name for the new sips-group
    # visit_of interest_2 = same as in visit of interest (psychs visits) but including also screenign here. 
    conversion_sips_group_date_scr = create_min_date('conversion_sips_group_date', df, df, onsetdate_sips_group, visit_of_interest, all_visits, 'str')
    conversion_sips_group_date_scr['conversion_sips_group_date'] = pd.to_datetime(conversion_sips_group_date_scr['value'])
    conversion_sips_group_date_scr = conversion_sips_group_date_scr[['redcap_event_name', 'conversion_sips_group_date']]
    conversion_date_scr_match = psychosis_onset_df.copy()
    conversion_date_scr_match['conversion_date']=conversion_date_scr_match['value']
    conversion_date_scr_match = conversion_date_scr_match[['redcap_event_name', 'conversion_date']]
    df_all_copy=df.copy()
    combined_vars = ['redcap_event_name', psychosis_str] + sips_vars_interest_screening
    df_all_copy=df_all_copy[combined_vars]
    sips_conv_vars = [psychosis_str] + sips_vars_interest_screening
    df_all_copy[sips_conv_vars] = df_all_copy[sips_conv_vars].astype(str)
    sips_merged = pd.merge(pd.merge(conversion_date_scr_match, conversion_sips_group_date_scr, on = 'redcap_event_name', how = 'left'),df_all_copy, on = 'redcap_event_name', how = 'left')
    sips_merged['sips_iv'] = np.where((sips_merged[sips_vars_interest_screening]=='1').any(axis=1), '1',\
                             np.where((sips_merged[sips_vars_interest_screening]=='0').all(axis=1), '0','-900'))
    sips_merged['conversion_date'] = pd.to_datetime(sips_merged['conversion_date'])
    sips_merged['psychs_fu_ac8_new']=np.where((sips_merged['sips_iv']=='1') & ((sips_merged[psychosis_str]=='0')|\
                                              (not any (date in ('1909-09-09', '1903-03-03') for date in sips_merged[['conversion_date', 'conversion_sips_group_date']])) &\
                                              (sips_merged['conversion_sips_group_date']<sips_merged['conversion_date'])), '1',\
                                     np.where((sips_merged['sips_iv']=='0')|\
                                             ((sips_merged[psychosis_str]=='1')&\
                                              (not any (date in ('1909-09-09', '1903-03-03') for date in sips_merged[['conversion_date', 'conversion_sips_group_date']])) &\
                                              (sips_merged['conversion_sips_group_date']>sips_merged['conversion_date'])), '0','-900'))
    sips_new_final = create_use_value(new_sips_group, sips_merged, df, ['psychs_fu_ac8_new'], visit_of_interest, all_visits, 'int')
    # create the overall lifetime bips variable
    return sips_new_final

def create_sips_groups(new_sips_group,sips_group_lifetime, df, df_sips_group_scr, onsetdate_sips_group, visit_of_interest, all_visits, conversion_df, sips_vars_interest_fu,\
                       conv_str, visit_of_interest_2):
    # create the different sips group assignments at follow-up
    # df: dataframe (df_all) including all variables
    # sips_vars_interest_scr: all screening variables needed to define baseline diagnosis
    # onsetdate_sips_group: onset date of the sips-group for all symptoms
    # visit_of_interest: the visits (here probably voi_8) that are of interest for the questionnaire (psychs)
    # all_visits: list with all visits (all_visits_list)
    # conversion_df: dataframe with the psychosis conversion date (conversion_date_fu)
    # sips_vars_interest_fu: vars of interest for the follow-up visits (e.g. bips symptoms)
    # conv_str = variable indicating psychosis conversion or not *psychs_fu_ac1_conv
    # new_sips_group = string iwth the name for the new sips-group
    # sips_group_lifetime = string with the name for the sips-group lifetime
    # visit_of interest_2 = same as in visit of interest (psychs visits) but including also screenign here. 
    sips_scr = df_sips_group_scr.copy()
    sips_vars_scr = ['redcap_event_name', 'value']
    sips_scr = sips_scr[sips_vars_scr]
    sips_scr['iv_scr']=np.where((sips_scr['value']==1), '1',\
                       np.where((sips_scr['value']==0), '0','-900'))
    sips_scr['value_1']=sips_scr['value']
    sips_scr=sips_scr[['redcap_event_name', 'iv_scr', 'value_1']]
    conversion_sips_group_date_fu = create_min_date('conversion_sips_group_date', df, df, onsetdate_sips_group, visit_of_interest, all_visits, 'str')
    conversion_sips_group_date_fu['conversion_sips_group_date'] = pd.to_datetime(conversion_sips_group_date_fu['value'])
    conversion_sips_group_date_fu = conversion_sips_group_date_fu[['redcap_event_name', 'conversion_sips_group_date']]
    conversion_date_fu_match = conversion_df.copy()
    conversion_date_fu_match['conversion_date']=conversion_date_fu_match['value']
    conversion_date_fu_match = conversion_date_fu_match[['redcap_event_name', 'conversion_date']]
    df_all_copy=df.copy()
    combined_vars = ['redcap_event_name', conv_str] + sips_vars_interest_fu
    df_all_copy=df_all_copy[combined_vars]
    sips_conv_vars = [conv_str] + sips_vars_interest_fu
    df_all_copy[sips_conv_vars] = df_all_copy[sips_conv_vars].astype(str)
    sips_merged = pd.merge(pd.merge(conversion_date_fu_match, conversion_sips_group_date_fu, on = 'redcap_event_name', how = 'left'),df_all_copy, on = 'redcap_event_name', how = 'left')
    sips_merged['sips_iv'] = np.where((sips_merged[sips_vars_interest_fu]=='1').any(axis=1), '1',\
                             np.where((sips_merged[sips_vars_interest_fu]=='0').all(axis=1), '0','-900'))
    sips_merged['conversion_date'] = pd.to_datetime(sips_merged['conversion_date'])
    sips_merged['psychs_fu_ac8_new']=np.where((sips_merged['sips_iv']=='1') & ((sips_merged[conv_str]=='0')|\
                                              (not any (date in ('1909-09-09', '1903-03-03') for date in sips_merged[['conversion_date', 'conversion_sips_group_date']])) &\
                                              (sips_merged['conversion_sips_group_date']<sips_merged['conversion_date'])), '1',\
                                     np.where((sips_merged['sips_iv']=='0')|\
                                             ((sips_merged[conv_str]=='1')&\
                                              (not any (date in ('1909-09-09', '1903-03-03') for date in sips_merged[['conversion_date', 'conversion_sips_group_date']])) &\
                                              (sips_merged['conversion_sips_group_date']>sips_merged['conversion_date'])), '0','-900'))
    sips_new_final = create_use_value(new_sips_group, sips_merged, df, ['psychs_fu_ac8_new'], visit_of_interest, all_visits, 'int')
    # create the overall lifetime bips variable
    sips_ac = pd.merge(sips_new_final, sips_scr, on = 'redcap_event_name', how = 'left')
    sips_ac['value'] = sips_ac['value'].astype(str)
    sips_ac['sips_iv_yesno'] = np.where((sips_ac['value'] == '1'), 1, 0)
    sips_ac['cumulative_sum'] = sips_ac['sips_iv_yesno'].shift(1).fillna(0).cumsum()
    sips_ac['sips_scr_yesno'] = np.where((sips_ac['iv_scr']=='1'), 1, 0)
    sips_ac['cumulative_sum_scr'] =sips_ac['sips_scr_yesno'].shift(1).fillna(0).cumsum()
    sips_ac['psychs_fu_ac8'] = np.where((sips_ac['cumulative_sum_scr'] > 0)|(sips_ac['cumulative_sum'] > 0)|(sips_ac['value']=='1')|(sips_ac['iv_scr']=='1'), '1', '0')
    sips_ac[['value', 'iv_scr', 'redcap_event_name', 'psychs_fu_ac8']] = sips_ac[['value', 'iv_scr', 'redcap_event_name', 'psychs_fu_ac8']].astype(str).apply(lambda x: x.str.strip())
    sips_ac['psychs_fu_ac8_final']=np.where(((((sips_ac['value']=='-900')|(sips_ac['value']=='-300'))&(sips_ac['value'] != '1'))&\
                                               (sips_ac['redcap_event_name']!='screening_arm_1')&(sips_ac['redcap_event_name'] !='screening_arm_2'))|\
                                             ((sips_ac['iv_scr']=='-900') & \
                                             ((sips_ac['redcap_event_name'] == 'screening_arm_1')|(sips_ac['redcap_event_name'] =='screening_arm_2'))),\
                                   '-900', sips_ac['psychs_fu_ac8'])
    sips_ac_final = create_use_value(sips_group_lifetime, sips_ac, df, ['psychs_fu_ac8_final'], visit_of_interest_2, all_visits, 'int')
    return sips_new_final, sips_ac_final

def create_scid5(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type, outcome_2):
    use_value_df = create_use_value(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type)
    use_value_df2 = create_use_value(outcome_2, df_1, df_2, outcome_2, visit_of_interest, all_visits, fill_type)
    use_value_df2['value2'] = use_value_df2['value']
    use_value_df2 = use_value_df2[['redcap_event_name', 'value2']]
    use_value_merged = pd.merge(use_value_df, use_value_df2, on = 'redcap_event_name', how = 'left')
    use_value_merged['value'] = np.where(use_value_merged['value'] == 3, 1, np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0,\
                                use_value_merged['value']))
    use_value_final=use_value_merged[['variable', 'redcap_event_name', 'value']]
    return use_value_final

def create_scid5_affective(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type, outcome_2, outcome_3):
    use_value_df = create_use_value(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type)
    use_value_df2 = create_use_value(outcome_2, df_1, df_2, outcome_2, visit_of_interest, all_visits, fill_type)
    use_value_df3 = create_use_value(outcome_3, df_1, df_2, outcome_3, visit_of_interest, all_visits, fill_type)
    use_value_df2['value2'] = use_value_df2['value']
    use_value_df2 = use_value_df2[['redcap_event_name', 'value2']]
    use_value_df3['value3'] = use_value_df3['value']
    use_value_df3 = use_value_df3[['redcap_event_name', 'value3']]
    use_value_merged = pd.merge(pd.merge(use_value_df, use_value_df2, on = 'redcap_event_name', how = 'left'), use_value_df3, on = 'redcap_event_name', how = 'left')
    if outcome == 'chrscid_szaff_bipolar':
        use_value_merged['value'] = np.where((use_value_merged['value'] == 3) & (use_value_merged['value3'] == 1), 1, \
                                    np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0, use_value_merged['value']))
    elif outcome == 'chrscid_szaff_depression':
        use_value_merged['value'] = np.where((use_value_merged['value'] == 3) & (use_value_merged['value3'] == 2), 1, \
                                    np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0, use_value_merged['value']))
    elif outcome == 'chrscid_szaff_unspecified':
        use_value_merged['value'] = np.where((use_value_merged['value'] == 3) & (use_value_merged['value3'] != 2) & (use_value_merged['value3'] != 1), 1, \
                                    np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0, use_value_merged['value']))
    use_value_final=use_value_merged[['variable', 'redcap_event_name', 'value']]
    return use_value_final

def create_scid5_bipolar_diff(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type, outcome_2, outcome_3):
    use_value_df = create_use_value(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type)
    use_value_df2 = create_use_value(outcome_2, df_1, df_2, outcome_2, visit_of_interest, all_visits, fill_type)
    use_value_df3 = create_use_value(outcome_3, df_1, df_2, outcome_3, visit_of_interest, all_visits, fill_type)
    use_value_df2['value2'] = use_value_df2['value']
    use_value_df2 = use_value_df2[['redcap_event_name', 'value2']]
    use_value_df3['value3'] = use_value_df3['value']
    use_value_df3 = use_value_df3[['redcap_event_name', 'value3']]
    use_value_merged = pd.merge(pd.merge(use_value_df, use_value_df2, on = 'redcap_event_name', how = 'left'), use_value_df3, on = 'redcap_event_name', how = 'left')
    if outcome == 'chrscid_bp_wo_psychosis':
        use_value_merged['value'] = np.where((use_value_merged['value'] == 3) & (use_value_merged['value3'] != 1), 1, \
                                    np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0, use_value_merged['value']))
    elif outcome == 'chrscid_bp_w_psychosis':
        use_value_merged['value'] = np.where((use_value_merged['value'] == 3) & (use_value_merged['value3'] == 1), 1, \
                                    np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0, use_value_merged['value']))
    use_value_final=use_value_merged[['variable', 'redcap_event_name', 'value']]
    return use_value_final

def create_scid5_mdd_diff(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type, outcome_2, outcome_3):
    use_value_df = create_use_value(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type)
    use_value_df2 = create_use_value(outcome_2, df_1, df_2, outcome_2, visit_of_interest, all_visits, fill_type)
    use_value_df3 = create_use_value(outcome_3, df_1, df_2, outcome_3, visit_of_interest, all_visits, fill_type)
    use_value_df2['value2'] = use_value_df2['value']
    use_value_df2 = use_value_df2[['redcap_event_name', 'value2']]
    use_value_df3['value3'] = use_value_df3['value']
    use_value_df3 = use_value_df3[['redcap_event_name', 'value3']]
    use_value_merged = pd.merge(pd.merge(use_value_df, use_value_df2, on = 'redcap_event_name', how = 'left'), use_value_df3, on = 'redcap_event_name', how = 'left')
    if outcome == 'chrscid_mdd_wo_psychosis':
        use_value_merged['value'] = np.where((use_value_merged['value'] == 3) & (use_value_merged['value3'] != 1), 1, \
                                    np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0, use_value_merged['value']))
    elif outcome == 'chrscid_mdd_w_psychosis':
        use_value_merged['value'] = np.where((use_value_merged['value'] == 3) & (use_value_merged['value3'] == 1), 1, \
                                    np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0, use_value_merged['value']))
    elif outcome == 'chrscid_persistent_depr':
        use_value_merged['value'] = np.where((use_value_merged['value'] == 3) | (use_value_merged['value3'] == 3), 1, \
                                    np.where((use_value_merged['value2'] == 1) | (use_value_merged['value2'] == 2) | (use_value_merged['value2'] == 3), 0, use_value_merged['value']))
    use_value_final=use_value_merged[['variable', 'redcap_event_name', 'value']]
    return use_value_final

def create_scid5_substance(outcome, df_1, df_2, var_list, visit_of_interest, all_visits, fill_type, outcome_1):
    value_max_substance = create_max_scid('max_substance', df_1, df_2, var_list, visit_of_interest, all_visits, 'int')
    value_max_substance['value_max'] = value_max_substance['value']
    value_max_substance = value_max_substance[['redcap_event_name', 'value_max']]
    use_value_df = create_use_value(outcome, df_1, df_2, [var_list[0]], visit_of_interest, all_visits, fill_type)
    use_value_df['value1'] = use_value_df['value']
    use_value_df =           use_value_df[['redcap_event_name', 'value1']]
    use_value_df2 = create_use_value(var_list[1], df_1, df_2, [var_list[1]], visit_of_interest, all_visits, fill_type)
    use_value_df2['value2'] = use_value_df2['value']
    use_value_df2 =           use_value_df2[['redcap_event_name', 'value2']]
    use_value_df3 = create_use_value(outcome_1, df_1, df_2, [outcome_1], visit_of_interest, all_visits, fill_type)
    use_value_df3['value3'] = use_value_df3['value']
    use_value_df3 =           use_value_df3[['redcap_event_name', 'value3']]
    sub_value = pd.merge(pd.merge(pd.merge(value_max_substance, use_value_df, on = 'redcap_event_name'), use_value_df2, on = 'redcap_event_name'), use_value_df3, on = 'redcap_event_name')
    sub_value['value'] = np.where(sub_value['value_max']>1, sub_value['value_max'], np.where((sub_value['value3'] == 1) | (sub_value['value3'] == 0), 0, -900))
    sub_value['value'] = np.where((sub_value['value'] == 0) | (sub_value['value'] == 1), 0, np.where((sub_value['value']==2) | (sub_value['value'] == 3), 1, \
                         np.where((sub_value['value'] == 4) | (sub_value['value'] == 5), 2, np.where(sub_value['value']>5, 3, sub_value['value']))))
    sub_value_final = create_use_value(outcome, sub_value, df_2, ['value'], visit_of_interest, all_visits, fill_type)
    return sub_value_final


def compute_outcomes(subject_id: str) -> Optional[pd.DataFrame]:
    '''create all warning messages as empty strings'''
    num_warnings = 11
    warnings = ['']*num_warnings
    # here I just create an example for how the warning messages could be printed:
    current_date_warn = datetime.now()
    # Format the date as a string
    formatted_date_warn = current_date_warn.strftime('%Y-%m-%d %H:%M:%S')
    
    # load the json data
    id = subject_id
    site=id[0:2]
    sub_data = '/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/{2}.{0}.json'.format(Network, site, id)
    if os.path.isfile(sub_data):
        print(f"File {sub_data} is present")
        df_all = pull_data(Network, id)
    else:
        print(f"File {sub_data} does not exist, skipping...")
        #warnings[0] = 'no file exist'
        return None
    # first create some important variables that you will need throughout the script
    # add a conversion of '.' because this is causing issues:
    def replace_dot(value):
        if value == '.':
            if isinstance(value, str):
                return '-900'
            else:
                return -900
        return value
    df_all = df_all.applymap(replace_dot)
    def replace_NslashA(value):
        if value == 'N/A':
            if isinstance(value, str):
                return '-900'
            else:
                return -900
        return value
    df_all = df_all.applymap(replace_NslashA)
    if df_all['redcap_event_name'].astype(str).str.contains('arm_1').any():
        group = 'chr'
    elif df_all['redcap_event_name'].astype(str).str.contains('arm_2').any():
        group = 'hc'
    baseln_df = df_all[df_all['redcap_event_name'].str.contains('basel')]
    if group == 'chr':
        age_1 = baseln_df['chrdemo_age_yrs_chr'].fillna(-900).to_numpy(dtype=float)
        age_2 = baseln_df['chrdemo_age_mos_chr'].fillna(-900).to_numpy(dtype=float)/12
        if len(age_2)>1:
            return None
        else:
            if age_2 < 0:
                age_2 = age_2*12
    elif group == 'hc':
        age_1 = baseln_df['chrdemo_age_yrs_hc'].fillna(-900).to_numpy(dtype=float)
        age_2 = baseln_df['chrdemo_age_mos_hc'].fillna(-900).to_numpy(dtype=float)/12
        if age_2 <0:
            age_2 = age_2 * 12
    else: 
        print(f"Subject is neither hc nor chr {id}")
        warnings[1] = f"Subject is neither hc nor chr {id}"
    age_3 = baseln_df['chrdemo_age_mos3'].fillna(-900).to_numpy(dtype=float)
    age_4 = baseln_df['chrdemo_age_mos2'].fillna(-900).to_numpy(dtype=float)/12
    if age_4 <0:
        age_4 = age_4 * 12
    if age_1 != -900 and age_1 != -3 and age_1 != -9:
        age = age_1 
    elif age_2 != -900 and age_2 != -3 and age_2 != -9:
        age = age_2
    elif age_3 != -900 and age_3 != -3 and age_3 != -9:
        age = age_3
    else:
        age = age_4
    if age.size == 0:
        age = -900
    if df_all['chrdemo_sexassigned'].astype(str).str.contains('1').any():
        sex = 'male'
    elif df_all['chrdemo_sexassigned'].astype(str).str.contains('2').any():
        sex = 'female'
    else:
        sex = 'unknown'
    baseln_df = df_all[df_all['redcap_event_name'].str.contains('basel')]
    df_all['chrpas_pmod_adult3v1'] = np.where(df_all['chrpas_pmod_adult3v1'] == '1909-09-09', -900, df_all['chrpas_pmod_adult3v1'])
    df_all['chrpas_pmod_adult3v1'] = np.where(df_all['chrpas_pmod_adult3v1'] == '1903-03-03', -300, df_all['chrpas_pmod_adult3v1'])
    df_all['chrpas_pmod_adult3v1'] = df_all['chrpas_pmod_adult3v1'].fillna(-900).astype(int)
    df_all['chrpas_pmod_adult3v1'] = np.where(df_all['chrpas_pmod_adult3v1'] == 9, -900, df_all['chrpas_pmod_adult3v1'])
    df_all['chrpas_pmod_adult3v3'] = np.where(df_all['chrpas_pmod_adult3v3'] == '1909-09-09', -900, df_all['chrpas_pmod_adult3v3'])
    df_all['chrpas_pmod_adult3v3'] = np.where(df_all['chrpas_pmod_adult3v3'] == '1903-03-03', -300, df_all['chrpas_pmod_adult3v3'])
    df_all['chrpas_pmod_adult3v3'] = df_all['chrpas_pmod_adult3v3'].fillna(-900).astype(int)
    df_all['chrpas_pmod_adult3v3'] = np.where(df_all['chrpas_pmod_adult3v3'] == 9, -900, df_all['chrpas_pmod_adult3v3'])
    month1_df = df_all[df_all['redcap_event_name'].str.contains('month_1_')]
    married_1 = np.nan_to_num(month1_df['chrpas_pmod_adult3v1'].to_numpy(dtype=int), nan = -900)
    married_1 = np.where(married_1.size == 0, -900, married_1)
    married_2 = np.nan_to_num(month1_df['chrpas_pmod_adult3v3'].to_numpy(dtype=int), nan = -900)
    married_2 = np.where(married_2.size == 0, -900, married_2)
    if married_1.size == 0:
        married_1 = -900
    if married_2.size == 0:
        married_2 = -900
# --------------------------------------------------------------------#
# We first extract/create some important variables for the script 
# --------------------------------------------------------------------#
# create the visits relevant for several variables:
    voi_0 = ''
    voi_1 = 'basel|month_1_arm_1|month_2_|month_3_arm_1|month_6_arm_1|month_12_arm_1|month_18_arm_1|month_24_arm_1|conversion_arm_1'
    voi_2 = 'basel'
    voi_3 = 'month_1_arm_1'
    voi_4 = 'basel|month_1_arm_1|month_2_|month_3_arm_1|month_4_arm_1|month_5_arm_1|month_6_arm_1|month_7_arm_1|month_8_arm_1|month_9_arm_1|month_10_arm_1|month_11_arm_1|month_12_arm_1|month_18_arm_1|month_24_arm_1|conversion_arm_1'
    voi_5 = 'basel|month_2_|month_6_|month_12_arm_1|month_18_arm_1|month_24_arm_1|conversion_arm_1'
    voi_6 = 'screening'
    voi_7 = 'basel|month_2_|month_6_arm_1|month_12_arm_1|month_18_arm_1|month_24_arm_1|conversion_arm_1'
    voi_8 = 'basel|month_1_arm_1|month_2_|month_3_arm_1|month_6_arm_1|month_12_|month_18_arm_1|month_24_|conversion_'
    voi_9 = 'screening|basel|month_1_|month_2_|month_3_|month_4_|month_5_|month_6|month_7_|month_8_|month_9|month_10_|month_11_|month_12_|month_18_|month_24_|conversion_|floating'
    voi_10= 'screening|basel|month_1_arm_1|month_2_|month_3_arm_1|month_6_arm_1|month_12_|month_18_arm_1|month_24_|conversion_'
    voi_11= 'basel|month_12_|month_24_|conversion_'
    voi_12= 'month_2_|month_6_arm_1|month_12_arm_1|month_18_arm_1|month_24_arm_1|conversion_'

# --------------------------------------------------------------------#
# start the functions
# --------------------------------------------------------------------#

# --------------------------------------------------------------------#
# CDSS 
# --------------------------------------------------------------------#
    cdss = create_total_division('chrcdss_total', df_all, df_all, ['chrcdss_calg1', 'chrcdss_calg2', 'chrcdss_calg3', 'chrcdss_calg4',\
                                                                'chrcdss_calg5', 'chrcdss_calg6', 'chrcdss_calg7', 'chrcdss_calg8', 'chrcdss_calg9'], 1, voi_1, all_visits_list, 'int')
    cdss['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# Perceived Discrimination Scale 
# --------------------------------------------------------------------#
    pdt = create_total_division('chrdim_perceived_discrim_total', df_all, df_all, ['chrdim_dim_yesno_q1_1','chrdlm_dim_yesno_q1_2','chrdlm_dim_sex','chrdlm_dim_yesno_age','chrdlm_dim_yesno_q4_1',\
                                                                                   'chrdlm_dim_yesno_q5','chrdlm_dim_yesno_q3','chrdlm_dim_yesno_q6','chrdlm_dim_yesno_other'], 1, voi_2, all_visits_list, 'int')
    pdt['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# OASIS 
# --------------------------------------------------------------------#
    oasis = create_total_division('chroasis_total', df_all, df_all, ['chroasis_oasis_1','chroasis_oasis_2', 'chroasis_oasis_3', 'chroasis_oasis_4', 'chroasis_oasis_5'], 1, voi_1, all_visits_list, 'int')
    oasis['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# Perceived Stress Scale 
# --------------------------------------------------------------------#
    pss_df = df_all.copy()
    pss_df['chrpss_pssp2_1']  = 4 - pss_df['chrpss_pssp2_1'].astype(float)
    pss_df['chrpss_pssp2_2']  = 4 - pss_df['chrpss_pssp2_2'].astype(float)
    pss_df['chrpss_pssp2_4']  = 4 - pss_df['chrpss_pssp2_4'].astype(float)
    pss_df['chrpss_pssp2_5']  = 4 - pss_df['chrpss_pssp2_5'].astype(float)
    #changed data minus
    pss = create_total_division('chrpss_perceived_stress_scale_total', pss_df, df_all, ['chrpss_pssp1_1','chrpss_pssp1_2', 'chrpss_pssp1_3','chrpss_pssp2_1', 'chrpss_pssp2_2','chrpss_pssp2_3',\
                                                                                'chrpss_pssp2_4','chrpss_pssp2_5', 'chrpss_pssp3_1','chrpss_pssp3_4'], 1, voi_1, all_visits_list, 'int')
    pss['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# BPRS 
# --------------------------------------------------------------------#
    bprs0 = create_use_value('chrbprs_bprs_total', df_all, df_all, ['chrbprs_bprs_total'], voi_4, all_visits_list, 'int')
    bprs1 = create_total_division('chrbprs_affect_subscale', df_all, df_all, ['chrbprs_bprs_depr', 'chrbprs_bprs_suic', 'chrbprs_bprs_guil'], 1, voi_4, all_visits_list, 'int')
    bprs2 = create_total_division('chrbprs_positive_symptom_subscale', df_all, df_all, ['chrbprs_bprs_unus', 'chrbprs_bprs_hall', 'chrbprs_bprs_susp'], 1, voi_4, all_visits_list, 'int')
    bprs3 = create_total_division('chrbprs_negative_symptom_subscale', df_all, df_all, ['chrbprs_bprs_blun', 'chrbprs_bprs_motr', 'chrbprs_bprs_emot'], 1, voi_4, all_visits_list, 'int')
    bprs4 = create_total_division('chrbprs_activation_subscale', df_all, df_all, ['chrbprs_bprs_exci', 'chrbprs_bprs_mohy', 'chrbprs_bprs_elat'], 1, voi_4, all_visits_list, 'int')
    bprs5 = create_total_division('chrbprs_disorganization_subscale', df_all, df_all, ['chrbprs_bprs_diso', 'chrbprs_bprs_conc', 'chrbprs_bprs_self'], 1, voi_4, all_visits_list, 'int')
    bprs = pd.concat([bprs0,bprs1, bprs2, bprs3, bprs4, bprs5], axis = 0)
    bprs['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# GF-R
# --------------------------------------------------------------------#
    gfr = create_decline('chrgfrs_global_role_decline', df_all, df_all, ['chrgfr_gf_role_high', 'chrgfr_gf_role_scole'], voi_2, all_visits_list, 'int')
    gfr['data_type'] = 'Integer'
    gfr['value'] = np.where((gfr['value'] < 0) & (gfr['value']> -15), -900, gfr['value'])
# --------------------------------------------------------------------#
# GF- S
# --------------------------------------------------------------------#
    gfs = create_decline('chrgfss_global_social_decline', df_all, df_all, ['chrgfs_gf_social_high', 'chrgfs_gf_social_scale'], voi_2, all_visits_list, 'int')
    gfs['data_type'] = 'Integer'
    gfs['value'] = np.where((gfs['value'] < 0) & (gfs['value']> -15), -900, gfs['value'])
# --------------------------------------------------------------------#
# Pubertal Developmental Scale 
# --------------------------------------------------------------------#
    if age > 18:
        pds_female  = create_condition_value('chrpds_total_score_female_sex', df_all, df_all, voi_2, all_visits_list, 'int', -300)
        pds_male    = create_condition_value('chrpds_total_score_male_sex', df_all, df_all, voi_2, all_visits_list, 'int', -300)
    elif age < 19 and sex == 'female':
        df_pds = df_all.copy()
        df_pds['chrpds_pds_2_p'] = np.where(df_pds['chrpds_pds_2_p'] == '1903-03-03', -300, df_pds['chrpds_pds_2_p'])
        df_pds['chrpds_pds_3_p'] = np.where(df_pds['chrpds_pds_3_p'] == '1903-03-03', -300, df_pds['chrpds_pds_3_p'])
        pds_female  = create_total_division('chrpds_total_score_female_sex',df_pds,df_pds,['chrpds_pds_1_p','chrpds_pds_2_p','chrpds_pds_3_p','chrpds_pds_f4_p','chrpds_pds_f5b_p'],1, voi_2, all_visits_list, 'int')
        pds_male    = create_condition_value('chrpds_total_score_male_sex', df_all, df_all, voi_2, all_visits_list, 'int', -300)
    elif age < 19 and sex == 'male':
        pds_male  = create_total_division('chrpds_total_score_male_sex',df_all,df_all,['chrpds_pds_1_p', 'chrpds_pds_2_p', 'chrpds_pds_3_p', 'chrpds_pds_m4_p', 'chrpds_pds_m5_p'], 1, voi_2, all_visits_list, 'int')
        pds_female    = create_condition_value('chrpds_total_score_female_sex', df_all, df_all, voi_2, all_visits_list, 'int', -300)
    elif sex != 'female' and sex != 'male':
        pds_female = create_condition_value('chrpds_total_score_female_sex', df_all, df_all, voi_2, all_visits_list, 'int', -900)
        pds_male = create_condition_value('chrpds_total_score_male_sex', df_all, df_all, voi_2, all_visits_list, 'int', -900)
    elif np.isnan(age) and sex == 'female':
        pds_female  = create_total_division('chrpds_total_score_female_sex',df_all,df_all,['chrpds_pds_1_p','chrpds_pds_2_p','chrpds_pds_3_p','chrpds_pds_f4_p','chrpds_pds_f5b_p'],1, voi_2, all_visits_list, 'int')
        pds_male    = create_condition_value('chrpds_total_score_male_sex', df_all, df_all, voi_2, all_visits_list, 'int', -300)
    elif np.isnan(age) and sex == 'male':
        pds_female  = create_condition_value('chrpds_total_score_female_sex', df_all, df_all, voi_2, all_visits_list, 'int', -300)
        pds_male  = create_total_division('chrpds_total_score_male_sex',df_all,df_all,['chrpds_pds_1_p', 'chrpds_pds_2_p', 'chrpds_pds_3_p', 'chrpds_pds_m4_p', 'chrpds_pds_m5_p'], 1, voi_2, all_visits_list, 'int')
    else:
        print(f"Something weird is going on with age {id}")
        warnings[2] = f"Something weird is going on with age and or sex {id}"
    # menarche
    if age > 18 or sex == 'male':
        pds_menarche = create_condition_value('chrpds_pds_f5b_p', df_all, df_all, voi_2, all_visits_list, 'int', -300)
    elif age < 19 and sex == 'female':
        pds_menarche = create_use_value('chrpds_pds_f5b_p', df_all, df_all, ['chrpds_pds_f5b_p'], voi_2, all_visits_list, 'int')
    else:
        pds_menarche = create_condition_value('chrpds_pds_f5b_p', df_all, df_all, voi_2, all_visits_list, 'int', -900)
    pds_final = pd.concat([pds_female, pds_male, pds_menarche], axis = 0)
    pds_final['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# SOFAS: Screening
# --------------------------------------------------------------------#
    sofas_1 = create_use_value('chrsofas_premorbid', df_all, df_all, ['chrsofas_premorbid'], voi_6, all_visits_list, 'int')
    sofas_2 = create_use_value('chrsofas_currscore12mo', df_all, df_all, ['chrsofas_currscore12mo'], voi_6, all_visits_list, 'int')
    sofas_3 = create_use_value('chrsofas_currscore', df_all, df_all, ['chrsofas_currscore'], voi_6, all_visits_list, 'int')
    sofas_4 = create_use_value('chrsofas_lowscore', df_all, df_all, ['chrsofas_lowscore'], voi_6, all_visits_list, 'int')
    sofas_screening = pd.concat([sofas_1, sofas_2, sofas_3, sofas_4], axis = 0)
    sofas_screening['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# SOFAS
# --------------------------------------------------------------------#
    df_sofas = df_all.copy()
    df_sofas['chrsofas_currscore_fu'] = np.where(df_sofas['chrsofas_currscore_fu'] == '1909-09-09', -900, df_sofas['chrsofas_currscore_fu'])
    df_sofas['chrsofas_currscore12mo_fu'] = np.where(df_sofas['chrsofas_currscore12mo_fu'] == '1909-09-09', -900, df_sofas['chrsofas_currscore12mo_fu'])
    sofas_5 = create_use_value('chrsofas_currscore_fu', df_sofas, df_sofas, ['chrsofas_currscore_fu'], voi_8, all_visits_list, 'int')
    sofas_6 = create_use_value('chrsofas_currscore12mo_fu', df_sofas, df_sofas, ['chrsofas_currscore12mo_fu'], voi_8, all_visits_list, 'int')
    sofas_fu = pd.concat([sofas_5, sofas_6], axis = 0)
    sofas_fu['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# NSI-PR 
# --------------------------------------------------------------------#
    nsipr_1 = create_total_division('chrnsipr_motivation_and_pleasure_dimension', df_all, df_all, ['chrnsipr_item1_rating', 'chrnsipr_item2_rating', 'chrnsipr_item3_rating', 'chrnsipr_item4_rating',\
                                                                                                'chrnsipr_item5_rating',  'chrnsipr_item6_rating', 'chrnsipr_item7_rating'], 7, voi_1, all_visits_list, 'float')
    nsipr_2 = create_total_division('chrnsipr_diminished_expression_dimension', df_all, df_all, ['chrnsipr_item8_rating', 'chrnsipr_item9_rating', 'chrnsipr_item10_rating',\
                                                                                            'chrnsipr_item11_rating'], 4, voi_1, all_visits_list, 'float')
    nsipr_3 = create_total_division('chrnsipr_avolition_domain', df_all, df_all, ['chrnsipr_item1_rating', 'chrnsipr_item2_rating'], 2, voi_1, all_visits_list, 'float')
    nsipr_4 = create_total_division('chrnsipr_asociality_domain', df_all, df_all, ['chrnsipr_item3_rating', 'chrnsipr_item4_rating', 'chrnsipr_item5_rating'], 3, voi_1, all_visits_list, 'float')
    nsipr_5 = create_total_division('chrnsipr_anhedonia_domain', df_all, df_all, ['chrnsipr_item6_rating', 'chrnsipr_item7_rating'], 2, voi_1, all_visits_list, 'float')
    nsipr_6 = create_total_division('chrnsipr_blunted_affect_domain', df_all, df_all, ['chrnsipr_item8_rating', 'chrnsipr_item9_rating', 'chrnsipr_item10_rating'], 3, voi_1, all_visits_list, 'float')
    nsipr_7 = create_use_value('chrnsipr_item11_rating', df_all, df_all, ['chrnsipr_item11_rating'], voi_1, all_visits_list, 'float')
    nsipr = pd.concat([nsipr_1, nsipr_2, nsipr_3, nsipr_4, nsipr_5, nsipr_6, nsipr_7], axis = 0)
    nsipr['data_type'] = 'Float'
# --------------------------------------------------------------------#
# Promis 
# --------------------------------------------------------------------#
    promis_df = df_all.copy()
    # difference in data minus
    promis_df['chrpromis_sleep20']     = 6 - promis_df['chrpromis_sleep20'].astype(float)
    promis_df['chrpromis_sleep44']     = 6 - promis_df['chrpromis_sleep44'].astype(float)
    promis_df['chrpromise_sleep108']   = 6 - promis_df['chrpromise_sleep108'].astype(float)
    promis_df['chrpromis_sleep72']     = 6 - promis_df['chrpromis_sleep72'].astype(float)
    promis_df['chrpromis_sleep67']     = 6 - promis_df['chrpromis_sleep67'].astype(float)
    promis = create_total_division('chrpromis_total', promis_df, df_all, ['chrpromis_sleep109','chrpromis_sleep116','chrpromis_sleep20','chrpromis_sleep44','chrpromise_sleep108','chrpromis_sleep72',\
                                                                    'chrpromis_sleep67','chrpromis_sleep115'], 1, voi_7, all_visits_list, 'int')
    promis['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# PGI_S
# --------------------------------------------------------------------#
    pgi_s = create_use_value('chrpgi_2', df_all, df_all, ['chrpgi_2'], voi_1, all_visits_list, 'int')
    pgi_s['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# RA-prediction
# --------------------------------------------------------------------#
    ra = create_use_value('chrpred_transition', df_all, df_all, ['chrpred_transition'], voi_2, all_visits_list, 'int')
    ra_2 = create_use_value('chrpred_experience', df_all, df_all, ['chrpred_experience'], voi_2, all_visits_list, 'int')
    ra = pd.concat([ra, ra_2], axis = 0)
    ra['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# CSSRS-baseline
# --------------------------------------------------------------------#
    if group == 'chr':
        df_pps = df_all[df_all['redcap_event_name'].str.contains('baseline_arm_1')]
    elif group == 'hc':
        df_pps = df_all[df_all['redcap_event_name'].str.contains('baseline_arm_2')]
    else: 
        print(f"Something werid is going on with hte group")
        warnings[3] = f"Something werid is going on with the group line 832"
    cssrs_sil_sum = df_pps[['chrcssrsb_si1l', 'chrcssrsb_si2l']].fillna(-900).astype(int).sum(axis = 1).to_numpy(dtype=int)
    cssrs_sim_sum = df_pps[['chrcssrsb_css_sim1', 'chrcssrsb_css_sim2']].fillna(-900).astype(int).sum(axis = 1).to_numpy(dtype=int)
    # In the week from April, 16th - April 22nd we have decided (Sylvain and Cheryl) to give a non-applicable instead of 0 if individuals never had any suicidal ideation
    if cssrs_sil_sum == 4:
        cssrs1 = create_condition_value('chrcssrsb_intensity_lifetime', df_all, df_all, voi_2, all_visits_list, 'int', -300)
    else:
        cssrs1 = create_total_division('chrcssrsb_intensity_lifetime' , df_all, df_all, ['chrcssrsb_sidfrql','chrcssrsb_siddurl','chrcssrsb_sidctrl','chrcssrsb_siddtrl','chrcssrsb_sidrsnl'],\
                                    1, voi_2, all_visits_list, 'int')
    if cssrs_sil_sum == 4 or cssrs_sim_sum == 4:
        cssrs2 = create_condition_value('chrcssrsb_intensity_pastmonth', df_all, df_all, voi_2, all_visits_list, 'int', -300)
    else:
        cssrs2 = create_total_division('chrcssrsb_intensity_pastmonth', df_all, df_all, ['chrcssrsb_css_sipmfreq','chrcssrsb_css_sipmdur','chrcssrsb_css_sipmctrl','chrcssrsb_css_sipmdet','chrcssrsb_css_sipmreas'],\
                                    1, voi_2, all_visits_list, 'int')
    cssrs = pd.concat([cssrs1, cssrs2], axis = 0)
    cssrs['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# CSSRS-Follow-up
# --------------------------------------------------------------------#
    cssrs_fu = df_all.copy()
    cssrs_fu['cssrs_fu_sim_sum'] = cssrs_fu[['chrcssrsfu_css_sim1', 'chrcssrsfu_css_sim2']].fillna(-900).astype(int).sum(axis = 1).to_numpy(dtype=int)
    cssrs_fu = cssrs_fu[['redcap_event_name', 'cssrs_fu_sim_sum']]
    cssrs_fu_first = create_total_division('chrcssrsfu_int_since_past', df_all, df_all, ['chrcssrsfu_css_sipmfreq', 'chrcssrsfu_css_sipmdur', 'chrcssrsfu_css_sipmctrl', 'chrcssrsfu_css_sipmdet', 'chrcssrsfu_css_sipmreas'],\
                              1, voi_12, all_visits_list, 'int')
    cssrs_fu_both = pd.merge(cssrs_fu_first, cssrs_fu, on = 'redcap_event_name', how = 'left')
    cssrs_fu_both['value'] = np.where(cssrs_fu_both['cssrs_fu_sim_sum'] == 0, -300, cssrs_fu_both['value'])
    cssrs_fu_both = cssrs_fu_both.drop('cssrs_fu_sim_sum', axis = 1)
    cssrs_fu_both['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# Premorbid adjustment scale
# --------------------------------------------------------------------#
    pas_child1    = create_total_division('chrpas_childhood_subtotal' , df_all, df_all, ['chrpas_pmod_child1','chrpas_pmod_child2','chrpas_pmod_child3','chrpas_pmod_child4'], 24, voi_3, all_visits_list, 'float')
    pas_earlyadol = create_total_division('chrpas_early_adolescence_subtotal' , df_all, df_all, ['chrpas_pmod_adol_early1','chrpas_pmod_adol_early2','chrpas_pmod_adol_early3','chrpas_pmod_adol_early4',\
                                                                                        'chrpas_pmod_adol_early5'], 30, voi_3, all_visits_list, 'float')
    pas_lateadol  = create_total_division('chrpas_late_adolescence_subtotal' , df_all, df_all, ['chrpas_pmod_adol_late1','chrpas_pmod_adol_late2','chrpas_pmod_adol_late3','chrpas_pmod_adol_late4',\
                                                                                        'chrpas_pmod_adol_late5'], 30, voi_3, all_visits_list, 'float')
    # for pas-adult the value for N/A can be 9. This does not fit our coding of missing/applicable. Therefore we change it here.
    if (married_1 == 0) and (married_2 == 0):
        pas_adult = create_fake_df_float(['chrpas_adulthood_subtotal'], all_visits_list, voi_3)
    if (married_1 == -900 or married_1 == -9 or married_1 == -3) and (married_2 == -900 or married_2 == -9 or married_2 == -3):
        pas_adult = create_total_division('chrpas_adulthood_subtotal' , df_all, df_all, ['chrpas_pmod_adult1','chrpas_pmod_adult2','chrpas_pmod_adult3v1'], 18, voi_3, all_visits_list, 'float')
    elif married_2 == -900 or married_2 == -9 or married_2 == -3:
        pas_adult = create_total_division('chrpas_adulthood_subtotal' , df_all, df_all, ['chrpas_pmod_adult1','chrpas_pmod_adult2','chrpas_pmod_adult3v1'], 18, voi_3, all_visits_list, 'float')
    elif married_1 == -900 or married_1 == -9 or married_1 == -3:
        pas_adult = create_total_division('chrpas_adulthood_subtotal' , df_all, df_all, ['chrpas_pmod_adult1','chrpas_pmod_adult2','chrpas_pmod_adult3v3'], 18, voi_3, all_visits_list, 'float')
    else:
        print(f"Something odd is going on with the married variable {id}")
        warnings[4]=f"Something odd is going on with the married variable {id} line 867"
    pas_child_merge=pas_child1.copy()
    pas_earlyadol_merge=pas_earlyadol.copy()
    pas_lateadol_merge=pas_lateadol.copy()
    pas_adult_merge=pas_adult.copy()
    pas_child_merge['value_child']=pas_child_merge['value']
    pas_earlyadol_merge['value_early']=pas_earlyadol_merge['value']
    pas_lateadol_merge['value_late']=pas_lateadol_merge['value']
    pas_adult_merge['value_adult']=pas_adult_merge['value']
    pas_child_early = pd.merge(pas_child_merge, pas_earlyadol_merge, on = 'redcap_event_name')
    pas_child_early_late = pd.merge(pd.merge(pas_child_merge, pas_earlyadol_merge, on = 'redcap_event_name'), pas_lateadol_merge, on = 'redcap_event_name')
    pas_child_early_late_adult = pd.merge(pd.merge(pd.merge(pas_child_merge, pas_earlyadol_merge, on = 'redcap_event_name'), pas_lateadol_merge, on = 'redcap_event_name'),\
                                        pas_adult_merge, on = 'redcap_event_name')
    pas_child_total=create_total_division('chrpas_total_score_only_childhood',df_all,df_all,['chrpas_pmod_child1','chrpas_pmod_child2','chrpas_pmod_child3','chrpas_pmod_child4'],24, voi_3, all_visits_list, 'float')
    pas_total_upto_early = create_total_division('chrpas_total_score_upto_early_adolescence', pas_child_early, df_all, ['value_child','value_early'], 2, voi_3, all_visits_list, 'float')
    pas_total_upto_late = create_total_division('chrpas_total_score_upto_late_adolescence', pas_child_early_late, df_all, ['value_child','value_early', 'value_late'], 3, voi_3, all_visits_list, 'float')
    pas_total_upto_adult = create_total_division('chrpas_total_score_upto_adulthood', pas_child_early_late_adult, df_all, ['value_child','value_early','value_late','value_adult'],4,voi_3, all_visits_list, 'float')
    premorbid_adjustment = pd.concat([pas_child1, pas_earlyadol, pas_lateadol, pas_adult, pas_child_total, pas_total_upto_early, pas_total_upto_late, pas_total_upto_adult], axis = 0)
    premorbid_adjustment['data_type'] = 'Float'
# --------------------------------------------------------------------#
# ASSIST
# --------------------------------------------------------------------#
    tobacco = create_assist('chrassist_tobacco', df_all, df_all, 'chrassist_whoassist_use1', 'chrassist_whoassist_often1',\
                                                                                   ['chrassist_whoassist_often1', 'chrassist_whoassist_urge1','chrassist_whoassist_prob1','chrassist_whoassist_fail1',\
                                                                                    'chrassist_whoassist_concern1','chrassist_whoassist_control1'], 1, voi_7, all_visits_list, 'int')
    alcohol = create_assist('chrassist_alcohol', df_all, df_all, 'chrassist_whoassist_use2', 'chrassist_whoassist_often2',\
                                                                                   ['chrassist_whoassist_often2', 'chrassist_whoassist_urge2','chrassist_whoassist_prob2','chrassist_whoassist_fail2',\
                                                                                    'chrassist_whoassist_concern2','chrassist_whoassist_control2'], 1, voi_7, all_visits_list, 'int')
    cannabis=create_assist('chrassist_cannabis', df_all, df_all, 'chrassist_whoassist_use3', 'chrassist_whoassist_often3',\
                                                                                   ['chrassist_whoassist_often3', 'chrassist_whoassist_urge3','chrassist_whoassist_prob3','chrassist_whoassist_fail3',\
                                                                                    'chrassist_whoassist_concern3','chrassist_whoassist_control3'], 1, voi_7, all_visits_list, 'int')
    cocaine = create_assist('chrassist_cocaine', df_all, df_all, 'chrassist_whoassist_use4', 'chrassist_whoassist_often4',\
                                                                                   ['chrassist_whoassist_often4', 'chrassist_whoassist_urge4','chrassist_whoassist_prob4','chrassist_whoassist_fail4',\
                                                                                    'chrassist_whoassist_concern4','chrassist_whoassist_control4'], 1, voi_7, all_visits_list, 'int')
    amphetamines=create_assist('chrassist_amphetamines',df_all, df_all, 'chrassist_whoassist_use5', 'chrassist_whoassist_often5',\
                                                                                   ['chrassist_whoassist_often5', 'chrassist_whoassist_urge5','chrassist_whoassist_prob5','chrassist_whoassist_fail5',\
                                                                                    'chrassist_whoassist_concern5','chrassist_whoassist_control5'], 1, voi_7, all_visits_list, 'int')
    inhalants = create_assist('chrassist_inhalants', df_all, df_all, 'chrassist_whoassist_use6',  'chrassist_whoassist_often6',\
                                                                                   ['chrassist_whoassist_often6', 'chrassist_whoassist_urge6','chrassist_whoassist_prob6','chrassist_whoassist_fail6',\
                                                                                    'chrassist_whoassist_concern6','chrassist_whoassist_control6'], 1, voi_7, all_visits_list, 'int')
    sedatives = create_assist('chrassist_sedatives', df_all, df_all, 'chrassist_whoassist_use7',  'chrassist_whoassist_often7',\
                                                                                   ['chrassist_whoassist_often7', 'chrassist_whoassist_urge7','chrassist_whoassist_prob7','chrassist_whoassist_fail7',\
                                                                                    'chrassist_whoassist_concern7','chrassist_whoassist_control7'], 1, voi_7, all_visits_list, 'int')
    hallucinogens=create_assist('chrassist_hallucinogens',df_all,df_all, 'chrassist_whoassist_use8',  'chrassist_whoassist_often8',\
                                                                                   ['chrassist_whoassist_often8', 'chrassist_whoassist_urge8','chrassist_whoassist_prob8','chrassist_whoassist_fail8',\
                                                                                    'chrassist_whoassist_concern8','chrassist_whoassist_control8'], 1, voi_7, all_visits_list, 'int')
    opiods = create_assist('chrassist_opiods', df_all, df_all, 'chrassist_whoassist_use9',  'chrassist_whoassist_often9',\
                                                                                   ['chrassist_whoassist_often9', 'chrassist_whoassist_urge9','chrassist_whoassist_prob9','chrassist_whoassist_fail9',\
                                                                                    'chrassist_whoassist_concern9','chrassist_whoassist_control9'], 1, voi_7, all_visits_list, 'int')
    other = create_assist('chrassist_other', df_all, df_all, 'chrassist_whoassist_use10', 'chrassist_whoassist_often10',\
                                                                                   ['chrassist_whoassist_often10', 'chrassist_whoassist_urge10','chrassist_whoassist_prob10','chrassist_whoassist_fail10',\
                                                                                    'chrassist_whoassist_concern10','chrassist_whoassist_control10'], 1, voi_7, all_visits_list, 'int')
    assist = pd.concat([tobacco, alcohol, cannabis, cocaine, amphetamines, inhalants, sedatives, hallucinogens, opiods, other], axis = 0)
    assist['data_type'] = 'Integer'
# --------------------------------------------------------------------#
# Polyenvironmental risk factor
# --------------------------------------------------------------------#
    if group == 'chr':
        df_figs = df_all[df_all['redcap_event_name'].str.contains('screening_arm_1')]
    elif group == 'hc':
        df_figs = df_all[df_all['redcap_event_name'].str.contains('screening_arm_2')]
    else:
        print(f"neither chr nor hc")
        warnings[5] = f"neither chr nor hc line 930"
    # pps based on age and gender:
    if age > 24 and age < 36 and sex == 'male':
        chrpps_sum1 = create_condition_value('chrpps_sum1', df_all, df_all, voi_2, all_visits_list, 'float', 2)
    elif (age > 0 and age < 25) or age > 36 or sex == 'female':
        chrpps_sum1 = create_condition_value('chrpps_sum1', df_all, df_all, voi_2, all_visits_list, 'float', 0)
    elif age == -900 or -9 or sex == 'unknown':
        chrpps_sum1 = create_condition_value('chrpps_sum1', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    elif age == -300 or -3:
        chrpps_sum1 = create_condition_value('chrpps_sum1', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    else:
        chrpps_sum1 = create_condition_value('chrpps_sum1', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    # pps 2 handedness
    total_handedness_df = create_total_division('hand', df_all, df_all, ['chrpps_writing','chrpps_throwing','chrpps_toothbrush','chrpps_spoon'],1, voi_2, all_visits_list, 'float')
    if group == 'chr':
        total_handedness_df = total_handedness_df[total_handedness_df['redcap_event_name'].str.contains('baseline_arm_1')]
        total_handedness = total_handedness_df['value'].to_numpy(dtype=float)
    elif group == 'hc':
        total_handedness_df = total_handedness_df[total_handedness_df['redcap_event_name'].str.contains('baseline_arm_2')]
        total_handedness = total_handedness_df['value'].to_numpy(dtype=float)
    else:
        warnings[6] = f"neither chr nor hc line 951"
    if total_handedness > 0 and total_handedness < 16:
        chrpps_sum2 = create_condition_value('chrpps_sum2', df_all, df_all, voi_2, all_visits_list, 'float', 2)
    elif total_handedness == -300:
        chrpps_sum2 = create_condition_value('chrpps_sum2', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    elif total_handedness == -900:
        chrpps_sum2 = create_condition_value('chrpps_sum2', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    else:
        chrpps_sum2 = create_condition_value('chrpps_sum2', df_all, df_all, voi_2, all_visits_list, 'float', 0)
    # pps 6 immigration
    if group == 'chr':
        pps6_df = df_all[df_all['redcap_event_name'].str.contains('baseline_arm_1')]
    elif group == 'hc':
        pps6_df = df_all[df_all['redcap_event_name'].str.contains('baseline_arm_2')]
    pps6_df.loc[:,'chrdemo_cob'] = pd.to_numeric(pps6_df['chrdemo_cob'], errors='coerce')
    pps6_df.loc[:,'chrdemo_country_situation'] = pd.to_numeric(pps6_df['chrdemo_country_situation'], errors='coerce')
    pps6_df.loc[:,'chrpps_mcob'] = pd.to_numeric(pps6_df['chrpps_mcob'], errors='coerce')
    pps6_df.loc[:,'chrpps_fcob'] = pd.to_numeric(pps6_df['chrpps_fcob'], errors='coerce')
    if pps6_df['chrdemo_country_situation'].isin([2,3]).any() & \
        pps6_df['chrdemo_cob'].isin([3,98,109,117,179]).any():
        chrpps_sum6 = create_condition_value('chrpps_sum6', df_all, df_all, voi_2, all_visits_list, 'float', 3)
    elif pps6_df['chrdemo_country_situation'].isin([2,3]).any() & \
        ~pps6_df['chrdemo_cob'].isin([3,98,109,117,179]).any():
        chrpps_sum6 = create_condition_value('chrpps_sum6', df_all, df_all, voi_2, all_visits_list, 'float', 2)
    elif pps6_df['chrdemo_country_situation'].isin([4]).any() & \
        (pps6_df['chrpps_fcob'].isin([3,98,109,117,179]).any() or \
        pps6_df['chrpps_mcob'].isin([3,98,109,117,179]).any()):
        chrpps_sum6 = create_condition_value('chrpps_sum6', df_all, df_all, voi_2, all_visits_list, 'float', 2.5)
    elif pps6_df['chrdemo_country_situation'].isin([4]).any() & \
        ~pps6_df['chrpps_fcob'].isin([3,98,109,117,179]).any() & \
        ~pps6_df['chrpps_mcob'].isin([3,98,109,117,179]).any():
        chrpps_sum6 = create_condition_value('chrpps_sum6', df_all, df_all, voi_2, all_visits_list, 'float', 1.5)
    elif pps6_df['chrdemo_country_situation'].isna().any() or pps6_df['chrdemo_country_situation'].isin([-900, -9, -99]).any():
        chrpps_sum6 = create_condition_value('chrpps_sum6', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    else:
        chrpps_sum6 = create_condition_value('chrpps_sum6', df_all, df_all, voi_2, all_visits_list, 'float', -0.5)
    # pps 7 paternal age
    #paternal_age_date = df_pps['chrpps_fdobpii'].astype(str).str.contains('1903-03-03')
    #print(list(df_all.filter(like='fdob').columns))
    # I have changed the paternal age calculation slightly because of newly introduced missing codes in date format
    #print(df_pps['chrpps_fdob'])
    df_pps['chrpps_fage'] = np.where(df_pps['chrpps_fage'] == '1909-09-09', -900, df_pps['chrpps_fage'])
    df_pps['chrpps_fage'] = np.where(df_pps['chrpps_fage'] == '1903-03-03', -300, df_pps['chrpps_fage'])
    paternal_age = df_pps['chrpps_fage'].fillna(-900).to_numpy(dtype=float)
    paternal_age_calc = paternal_age - age
    if paternal_age == -900 or paternal_age == -9 or age == -900:
        chrpps_sum7 = create_condition_value('chrpps_sum7', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    elif paternal_age == -300 or paternal_age == -3 or age == -300:
        chrpps_sum7 = create_condition_value('chrpps_sum7', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    elif paternal_age_calc > 45:
        chrpps_sum7 = create_condition_value('chrpps_sum7', df_all, df_all, voi_2, all_visits_list, 'float', 3.5)
    elif paternal_age_calc > 35:
        chrpps_sum7 = create_condition_value('chrpps_sum7', df_all, df_all, voi_2, all_visits_list, 'float', 0.5)
    else:
        chrpps_sum7 = create_condition_value('chrpps_sum7', df_all, df_all, voi_2, all_visits_list, 'float', -0.5)
    # pps 8 SES
    df_pps['chrpps_focc'] = np.where(df_pps['chrpps_focc'] == '1909-09-09', -900, df_pps['chrpps_focc'])
    df_pps['chrpps_focc'] = np.where(df_pps['chrpps_focc'] == '1903-03-03', -300, df_pps['chrpps_focc'])
    focc = df_pps['chrpps_focc'].fillna(-900).to_numpy(dtype=float)
    if focc == -900 or focc == -9:
        chrpps_sum8 = create_condition_value('chrpps_sum8', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    elif focc == -300 or focc == -3:
        chrpps_sum8 = create_condition_value('chrpps_sum8', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    elif focc > 6:
        chrpps_sum8 = create_condition_value('chrpps_sum8', df_all, df_all, voi_2, all_visits_list, 'float', 1)
    else:
        chrpps_sum8 = create_condition_value('chrpps_sum8', df_all, df_all, voi_2, all_visits_list, 'float', 0)
    # pps 9 family history of disorders 
    mother_ddx = df_figs['chrfigs_mother_ddx'].fillna(-900).to_numpy(dtype=float)
    mother_mdx = df_figs['chrfigs_mother_mdx'].fillna(-900).to_numpy(dtype=float)
    mother_pdx = df_figs['chrfigs_mother_pdx'].fillna(-900).to_numpy(dtype=float)
    mother_napdx = df_figs['chrfigs_mother_napdx'].fillna(-900).to_numpy(dtype=float)
    mother_apdx = df_figs['chrfigs_mother_apdx'].fillna(-900).to_numpy(dtype=float)
    father_ddx = df_figs['chrfigs_father_ddx'].fillna(-900).to_numpy(dtype=float)
    father_mdx = df_figs['chrfigs_father_mdx'].fillna(-900).to_numpy(dtype=float)
    father_pdx = df_figs['chrfigs_father_pdx'].fillna(-900).to_numpy(dtype=float)
    father_napdx = df_figs['chrfigs_father_napdx'].fillna(-900).to_numpy(dtype=float)
    father_apdx = df_figs['chrfigs_father_apdx'].fillna(-900).to_numpy(dtype=float)
    if mother_ddx == 3 or mother_mdx == 3 or mother_pdx == 3 or mother_napdx == 3 or mother_apdx == 3 or\
       father_ddx == 3 or father_mdx == 3 or father_pdx == 3 or father_napdx == 3 or father_apdx == 3:
        chrpps_sum9 = create_condition_value('chrpps_sum9', df_all, df_all, voi_2, all_visits_list, 'float', 5.5)
    elif (mother_ddx == 2 or mother_ddx == 1 or mother_ddx == 0) and (mother_mdx == 2 or mother_mdx == 1 or mother_mdx == 0) and\
         (mother_pdx == 2 or mother_pdx == 1 or mother_pdx == 0) and (mother_apdx == 2 or mother_apdx == 1 or mother_apdx == 0) and (mother_napdx == 2 or mother_napdx == 1 or mother_napdx == 0) and\
         (father_ddx == 2 or father_ddx == 1 or father_ddx == 0) and (father_mdx == 2 or father_mdx == 1 or father_mdx == 0) and\
         (father_pdx == 2 or father_pdx == 1 or father_pdx == 0) and (father_apdx == 2 or father_apdx == 1 or father_apdx == 0) and (father_napdx == 2 or father_napdx == 1 or father_napdx == 0):
        chrpps_sum9 = create_condition_value('chrpps_sum9', df_all, df_all, voi_2, all_visits_list, 'float', -2)
    if mother_ddx == -300 or mother_mdx == -300 or mother_pdx == -300 or mother_napdx == -300 or mother_apdx == -300 or\
       father_ddx == -300 or father_mdx == -300 or father_pdx == -300 or father_napdx == -300 or father_apdx == -300 or\
       mother_ddx == -3 or mother_mdx == -3 or mother_pdx == -3 or mother_napdx == -3 or mother_apdx == -3 or\
       father_ddx == -3 or father_mdx == -3 or father_pdx == -3 or father_napdx == -3 or father_apdx == -3:
        chrpps_sum9 = create_condition_value('chrpps_sum9', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    if mother_ddx == -900 or mother_mdx == -900 or mother_pdx == -900 or mother_napdx == -900 or mother_apdx == -900 or\
       father_ddx == -900 or father_mdx == -900 or father_pdx == -900 or father_napdx == -900 or father_apdx == -900 or\
       mother_ddx == -9 or mother_mdx == -9 or mother_pdx == -9 or mother_napdx == -9 or mother_apdx == -9 or\
       father_ddx == -9 or father_mdx == -9 or father_pdx == -9 or father_napdx == -9 or father_apdx == -9 or\
       mother_ddx == 9 or mother_mdx == 9 or mother_pdx == 9 or mother_napdx == 9 or mother_apdx == 9 or\
       father_ddx == 9 or father_mdx == 9 or father_pdx == 9 or father_napdx == 9 or father_apdx == 9:
        chrpps_sum9 = create_condition_value('chrpps_sum9', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    # I am not so sure about the chrpps_sum9: maybe I have to go back to this later. 
    # pps 10 life event
    sixmo_1  = df_pps['chrpps_sixmo___1'].fillna(-900).to_numpy(dtype=float)
    sixmo_2  = df_pps['chrpps_sixmo___2'].fillna(-900).to_numpy(dtype=float)
    sixmo_3  = df_pps['chrpps_sixmo___3'].fillna(-900).to_numpy(dtype=float)
    sixmo_4  = df_pps['chrpps_sixmo___4'].fillna(-900).to_numpy(dtype=float)
    sixmo_5  = df_pps['chrpps_sixmo___5'].fillna(-900).to_numpy(dtype=float)
    sixmo_6  = df_pps['chrpps_sixmo___6'].fillna(-900).to_numpy(dtype=float)
    sixmo_7  = df_pps['chrpps_sixmo___7'].fillna(-900).to_numpy(dtype=float)
    sixmo_8  = df_pps['chrpps_sixmo___8'].fillna(-900).to_numpy(dtype=float)
    sixmo_10 = df_pps['chrpps_sixmo___10'].fillna(-900).to_numpy(dtype=float)
    sixmo_11 = df_pps['chrpps_sixmo___11'].fillna(-900).to_numpy(dtype=float)
    sixmo_12 = df_pps['chrpps_sixmo___12'].fillna(-900).to_numpy(dtype=float)
    sixmo_13 = df_pps['chrpps_sixmo___13'].fillna(-900).to_numpy(dtype=float)
    sixmo_14 = df_pps['chrpps_sixmo___14'].fillna(-900).to_numpy(dtype=float)
    sixmo_15 = df_pps['chrpps_sixmo___15'].fillna(-900).to_numpy(dtype=float)
    if sixmo_1 == 1 or sixmo_2 == 1 or sixmo_3 == 1 or sixmo_4 == 1 or sixmo_5 == 1 or sixmo_6 == 1 or sixmo_7 == 1 or sixmo_8 == 1 or sixmo_10 == 1 or sixmo_11 == 1 or sixmo_12 == 1 or\
       sixmo_13 == 1 or sixmo_14 == 1:
        chrpps_sum10 = create_condition_value('chrpps_sum10', df_all, df_all, voi_2, all_visits_list, 'float', 5.5)
    elif sixmo_1 == 0 and sixmo_2 == 0 and sixmo_3 == 0 and sixmo_4 == 0 and sixmo_5 == 0 and sixmo_6 == 0 and sixmo_7 == 0 and sixmo_8 == 0 and sixmo_10 == 0 and sixmo_11 == 0 and sixmo_12 == 0 and\
         sixmo_13 == 0 and sixmo_14 == 0 and sixmo_15 == 1:
        chrpps_sum10 = create_condition_value('chrpps_sum10', df_all, df_all, voi_2, all_visits_list, 'float', -2)
    elif sixmo_1 == -900 or sixmo_2 == -900 or sixmo_3 == -900 or sixmo_4 == -900 or sixmo_5 == -900 or sixmo_6 == -900 or sixmo_7 == -900 or sixmo_8 == -900 or\
       sixmo_10 == -900 or sixmo_11 == -900 or sixmo_12 == -900 or sixmo_13 == -900 or sixmo_14 == -900:
        chrpps_sum10 = create_condition_value('chrpps_sum10', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    elif sixmo_1 == -300 or sixmo_2 == -300 or sixmo_3 == -300 or sixmo_4 == -300 or sixmo_5 == -300 or sixmo_6 == -300 or sixmo_7 == -300 or sixmo_8 == -300 or\
       sixmo_10 == -300 or sixmo_11 == -300 or sixmo_12 == -300 or sixmo_13 == -300 or sixmo_14 == -300:
        chrpps_sum10 = create_condition_value('chrpps_sum10', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    else:
        chrpps_sum10 = create_condition_value('chrpps_sum10', df_all, df_all, voi_2, all_visits_list, 'float', -900)
        #print('not sure what is going on with chrpps_sum10')
    # pps 11 tobacco
    assist_1  = df_pps['chrassist_whoassist_often1'].fillna(-900).to_numpy(dtype=float)
    assist_12  = df_pps['chrassist_whoassist_use1'].fillna(-900).to_numpy(dtype=float)
    if assist_1 == 6:
        chrpps_sum11 = create_condition_value('chrpps_sum11', df_all, df_all, voi_2, all_visits_list, 'float', 3)
    elif (assist_1 == -900 or assist_1 -9) and assist_12 != 0:
        chrpps_sum11 = create_condition_value('chrpps_sum11', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    elif (assist_1 == -300 or assist_1 == -3) and assist_12 != 0:
        chrpps_sum11 = create_condition_value('chrpps_sum11', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    else:
        chrpps_sum11 = create_condition_value('chrpps_sum11', df_all, df_all, voi_2, all_visits_list, 'float', -0.5)
    # pps 12 cannabis
    assist_2  = df_pps['chrassist_whoassist_often3'].fillna(-900).to_numpy(dtype=float)
    assist_22  = df_pps['chrassist_whoassist_use3'].fillna(-900).to_numpy(dtype=float)
    if assist_2 > 3:
        chrpps_sum12 = create_condition_value('chrpps_sum12', df_all, df_all, voi_2, all_visits_list, 'float', 7)
    elif (assist_2 == -900 or assist_2 == -9) and assist_22 != 0:
        chrpps_sum12 = create_condition_value('chrpps_sum12', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    elif (assist_2 == -300 or assist_2 == -3) and assist_22 != 0:
        chrpps_sum12 = create_condition_value('chrpps_sum12', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    else:
        chrpps_sum12 = create_condition_value('chrpps_sum12', df_all, df_all, voi_2, all_visits_list, 'float', 0)
    # pps 13 Childhood trauma
    ctq_df = df_all.copy()
    ctq_df['chrpps_special']   = 6 - ctq_df['chrpps_special'].astype(float)
    ctq_df['chrpps_care']      = 6 - ctq_df['chrpps_care'].astype(float)
    ctq_df['chrpps_loved']     = 6 - ctq_df['chrpps_loved'].astype(float)
    ctq_df['chrpps_closefam']  = 6 - ctq_df['chrpps_closefam'].astype(float)
    ctq_df['chrpps_support']   = 6 - ctq_df['chrpps_support'].astype(float)
    ctq_df['chrpps_protect']   = 6 - ctq_df['chrpps_protect'].astype(float)
    ctq_df['chrpps_docr']      = 6 - ctq_df['chrpps_docr'].astype(float)
    ctq = create_total_division('ctq', ctq_df, df_all, ['chrpps_lazy', 'chrpps_born', 'chrpps_hate', 'chrpps_hurt', 'chrpps_emoab', 'chrpps_doc', 'chrpps_bruise', 'chrpps_belt',\
                                                        'chrpps_physab', 'chrpps_beat', 'chrpps_touch', 'chrpps_threat', 'chrpps_sexual', 'chrpps_molest', 'chrpps_sexab', 'chrpps_loved',\
                                                        'chrpps_special', 'chrpps_care', 'chrpps_closefam', 'chrpps_support', 'chrpps_hunger', 'chrpps_protect', 'chrpps_pardrunk',\
                                                        'chrpps_dirty', 'chrpps_docr'], 1, voi_2, all_visits_list, 'float')
    if group == 'chr':
        ctq = ctq[ctq['redcap_event_name'].str.contains('baseline_arm_1')]
        ctq_final_score  = ctq['value'].to_numpy(dtype=float)
    elif group == 'hc':
        ctq = ctq[ctq['redcap_event_name'].str.contains('baseline_arm_2')]
        ctq_final_score  = ctq['value'].to_numpy(dtype=float)
    else:
        warnings[7] = f"neither chr nor hc 1094"
    if ctq_final_score > 55:
        chrpps_sum13 = create_condition_value('chrpps_sum13', df_all, df_all, voi_2, all_visits_list, 'float', 4)
    elif ctq_final_score > -1 and ctq_final_score < 56:
        chrpps_sum13 = create_condition_value('chrpps_sum13', df_all, df_all, voi_2, all_visits_list, 'float', -0.5)
    elif (ctq_final_score == -900 or ctq_final_score -9):
        chrpps_sum13 = create_condition_value('chrpps_sum13', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    elif (ctq_final_score == -300 or ctq_final_score == -3):
        chrpps_sum13 = create_condition_value('chrpps_sum13', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    else:
        prfloat("What is going on with the CTQ")
        warnings[8] = f"What is going on with the CTQ"
    # pps 14 Trait anhedonia
    trait_df = df_all.copy()
    trait_df['chrpps_restaur']   = 7 - trait_df['chrpps_restaur'].astype(float)
    trait = create_total_division('trait', trait_df, df_all, ['chrpps_taste', 'chrpps_restaur', 'chrpps_roller', 'chrpps_holiday', 'chrpps_tasty', 'chrpps_pleasure', 'chrpps_lookfwd',\
                                                              'chrpps_menu', 'chrpps_actor', 'chrpps_crackle', 'chrpps_rain', 'chrpps_grass', 'chrpps_air', 'chrpps_coffee', 'chrpps_hair',\
                                                              'chrpps_yawn', 'chrpps_snow'], 1, voi_2, all_visits_list, 'float')
    if group == 'chr':
        trait = trait[trait['redcap_event_name'].str.contains('baseline_arm_1')]
        trait_final_score  = trait['value'].to_numpy(dtype=float)
    elif group == 'hc':
        trait = trait[trait['redcap_event_name'].str.contains('baseline_arm_2')]
        trait_final_score  = trait['value'].to_numpy(dtype=float)
    else:
        warnings[9] = f"neither chr nor hc 1094"
    if trait_final_score < 36 and trait_final_score > -1:
        chrpps_sum14 = create_condition_value('chrpps_sum14', df_all, df_all, voi_2, all_visits_list, 'float', 6.5)
    elif trait_final_score > 35:
        chrpps_sum14 = create_condition_value('chrpps_sum14', df_all, df_all, voi_2, all_visits_list, 'float', 0)
    elif (trait_final_score == -900 or trait_final_score -9):
        chrpps_sum14 = create_condition_value('chrpps_sum14', df_all, df_all, voi_2, all_visits_list, 'float', -900)
    elif (trait_final_score == -300 or trait_final_score == -3):
        chrpps_sum14 = create_condition_value('chrpps_sum14', df_all, df_all, voi_2, all_visits_list, 'float', -300)
    else:
        warnings[10] = f"trait_final_score is werid 1129"
        # --------------------------------------------------------------------------
    # add the childhood trauma variables based on conversation with Luis Alameda
    ctq_traumageneral_df = create_total_division('chrpps_total', ctq_df, df_all, ['chrpps_lazy', 'chrpps_born', 'chrpps_hate', 'chrpps_hurt',\
                                                                                'chrpps_emoab', 'chrpps_doc', 'chrpps_bruise', 'chrpps_belt',\
                                                                                'chrpps_physab', 'chrpps_beat', 'chrpps_touch', 'chrpps_threat',\
                                                                                'chrpps_sexual', 'chrpps_molest', 'chrpps_sexab', 'chrpps_loved',\
                                                                                'chrpps_special', 'chrpps_care', 'chrpps_closefam', 'chrpps_support',\
                                                                                'chrpps_hunger', 'chrpps_protect', 'chrpps_pardrunk',\
                                                                                'chrpps_dirty', 'chrpps_docr'], 1, voi_2, all_visits_list, 'float')
    ctq_PA_df = create_total_division('chrpps_pa_sum', ctq_df, df_all, ['chrpps_doc', 'chrpps_bruise', 'chrpps_belt',\
                                                                                'chrpps_physab', 'chrpps_beat'], 1, voi_2, all_visits_list, 'float')
    ctq_SA_df = create_total_division('chrpps_sa_sum', ctq_df, df_all, ['chrpps_touch', 'chrpps_sexual', 'chrpps_molest',\
                                                                              'chrpps_sexab', 'chrpps_threat'], 1, voi_2, all_visits_list, 'float')
    ctq_EA_df = create_total_division('chrpps_ea_sum', ctq_df, df_all, ['chrpps_hurt', 'chrpps_hate', 'chrpps_emoab',\
                                                                                 'chrpps_lazy', 'chrpps_born'], 1, voi_2, all_visits_list, 'float')
    ctq_EN_df = create_total_division('chrpps_en_sum', ctq_df, df_all, ['chrpps_special', 'chrpps_loved', 'chrpps_care',\
                                                                                   'chrpps_closefam', 'chrpps_support'], 1, voi_2, all_visits_list, 'float')
    ctq_PN_df = create_total_division('chrpps_pn_sum', ctq_df, df_all, ['chrpps_hunger', 'chrpps_pardrunk', 'chrpps_dirty',\
                                                                                  'chrpps_docr', 'chrpps_protect'], 1, voi_2, all_visits_list, 'float')
    polyrisk = pd.concat([chrpps_sum1, chrpps_sum2, chrpps_sum6, chrpps_sum7, chrpps_sum8, chrpps_sum9, chrpps_sum10, chrpps_sum11, chrpps_sum12, chrpps_sum13, chrpps_sum14,\
                              ctq_traumageneral_df, ctq_PA_df, ctq_SA_df, ctq_EA_df, ctq_EN_df, ctq_PN_df], axis = 0)
    polyrisk['data_type'] = 'Float'
    #polyrisk = pd.concat([chrpps_sum1, chrpps_sum2, chrpps_sum7, chrpps_sum8, chrpps_sum9, chrpps_sum10, chrpps_sum11, chrpps_sum12, chrpps_sum13, chrpps_sum14], axis = 0)
# --------------------------------------------------------------------#
# SCID-V psychosis, depression, bipolar disorder, substance use
# --------------------------------------------------------------------#
    # psychosis
    scid5_delusional = create_scid5('chrscid_delusional', df_all, df_all, ['chrscid_c37'], voi_11, all_visits_list, 'int', ['chrscid_b1'])
    scid5_brief_psychotic = create_scid5('chrscid_briefpsychotic', df_all, df_all, ['chrscid_c44'], voi_11, all_visits_list, 'int', ['chrscid_b1'])
    scid5_schizophreniform = create_scid5('chrscid_szform', df_all, df_all, ['chrscid_c14'], voi_11, all_visits_list, 'int', ['chrscid_b1'])
    scid5_schizophrenia = create_scid5('chrscid_sz', df_all, df_all, ['chrscid_c10'], voi_11, all_visits_list, 'int', ['chrscid_b1'])
    scid5_schizoaffective_bipolar= create_scid5_affective('chrscid_szaff_bipolar', df_all, df_all, ['chrscid_c26'], voi_11, all_visits_list, 'int', ['chrscid_b1'], ['chrscid_c27'])
    scid5_schizoaffective_depression= create_scid5_affective('chrscid_szaff_depression', df_all, df_all, ['chrscid_c26'], voi_11, all_visits_list, 'int', ['chrscid_b1'], ['chrscid_c27'])
    scid5_schizoaffective_unspecified= create_scid5_affective('chrscid_szaff_unspecified', df_all, df_all, ['chrscid_c26'], voi_11, all_visits_list, 'int', ['chrscid_b1'], ['chrscid_c27'])
    scid5_subst_med_psychosis= create_scid5('chrscid_subst_med_psychosis', df_all, df_all, ['chrscid_c78'], voi_11, all_visits_list, 'int', ['chrscid_b1'])
    scid5_amc_psychosis= create_scid5('chrscid_amc_psychosis', df_all, df_all, ['chrscid_c71'], voi_11, all_visits_list, 'int', ['chrscid_b1'])
    scid5_other_psychosis= create_scid5('chrscid_other_psychosis', df_all, df_all, ['chrscid_c50'], voi_11, all_visits_list, 'int', ['chrscid_b1'])
    # bipolar disorder
    scid5_bp_wo_psychosis = create_scid5_bipolar_diff('chrscid_bp_wo_psychosis', df_all, df_all, ['chrscid_d3'], voi_11, all_visits_list, 'int', ['chrscid_a1'], ['chrscid_d47_d52___1'])
    scid5_bp_w_psychosis = create_scid5_bipolar_diff('chrscid_bp_w_psychosis', df_all, df_all, ['chrscid_d3'], voi_11, all_visits_list, 'int', ['chrscid_a1'], ['chrscid_d47_d52___1'])
    scid5_bp_2 = create_scid5('chrscid_bp_2', df_all, df_all, ['chrscid_d9'], voi_11, all_visits_list, 'int', ['chrscid_a1'])
    scid5_cyclothymic = create_scid5('chrscid_cyclothymic', df_all, df_all, ['chrscid_a138'], voi_11, all_visits_list, 'int', ['chrscid_a1'])
    scid5_subst_med_bp = create_scid5('chrscid_subst_med_bp', df_all, df_all, ['chrscid_a206'], voi_11, all_visits_list, 'int', ['chrscid_a1'])
    scid5_other_bp = create_scid5('chrscid_other_bp', df_all, df_all, ['chrscid_d23'], voi_11, all_visits_list, 'int', ['chrscid_a1'])
    # depressive disorders
    scid5_mdd_wo_psychosis = create_scid5_mdd_diff('chrscid_mdd_wo_psychosis', df_all, df_all, ['chrscid_d28'], voi_11, all_visits_list, 'int', ['chrscid_a1'], ['chrscid_d63___1'])
    scid5_mdd_w_psychosis = create_scid5_mdd_diff('chrscid_mdd_w_psychosis', df_all, df_all, ['chrscid_d28'], voi_11, all_visits_list, 'int', ['chrscid_a1'], ['chrscid_d63___1'])
    scid5_persistent_depr = create_scid5_mdd_diff('chrscid_persistent_depr', df_all, df_all, ['chrscid_a153'], voi_11, all_visits_list, 'int', ['chrscid_a1'], ['chrscid_a170'])
    scid5_subst_med_depr = create_scid5('chrscid_subst_med_depr', df_all, df_all, ['chrscid_a221'], voi_11, all_visits_list, 'int', ['chrscid_a1'])
    scid5_amc_depr = create_scid5('chrscid_amc_depr', df_all, df_all, ['chrscid_a213'], voi_11, all_visits_list, 'int', ['chrscid_a1'])
    scid5_other_depr = create_scid5('chrscid_other_depr', df_all, df_all, ['chrscid_d39'], voi_11, all_visits_list, 'int', ['chrscid_a1'])
    # building summary scores for the different disorders
    scid5_all_psychosis_condition= pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(scid5_delusional,scid5_brief_psychotic, on = 'redcap_event_name'),\
                                                                                                                                      scid5_schizophreniform, on = 'redcap_event_name'),\
                                                                                                                                      scid5_schizophrenia, on = 'redcap_event_name'), \
                                                                                                                                      scid5_schizoaffective_bipolar, on = 'redcap_event_name'),\
                                                                                                                                      scid5_schizoaffective_depression, on = 'redcap_event_name'), \
                                                                                                                                      scid5_schizoaffective_unspecified, on = 'redcap_event_name'),\
                                                                                                                                      scid5_subst_med_psychosis, on = 'redcap_event_name'), \
                                                                                                                                      scid5_amc_psychosis, on = 'redcap_event_name'), \
                                                                                                                                      scid5_other_psychosis, on = 'redcap_event_name'), \
                                                                                                                                      scid5_bp_w_psychosis, on = 'redcap_event_name'),\
                                                                                                                                      scid5_mdd_w_psychosis, on = 'redcap_event_name')
    scid5_all_psychosis_condition['value'] = np.where((scid5_all_psychosis_condition.filter(like='value').eq(1)).any(axis=1), 1,\
                                             np.where((scid5_all_psychosis_condition.filter(like='value').eq(0)).all(axis=1), 0, -900))
    scid5_all_psychosis = create_use_value('chrscid_any_psychosis', scid5_all_psychosis_condition, df_all, ['value'], voi_11, all_visits_list, 'int')
    scid5_all_bipolar_condition= pd.merge(pd.merge(pd.merge(pd.merge(scid5_bp_wo_psychosis,scid5_bp_2, on = 'redcap_event_name'),\
                                                                     scid5_cyclothymic, on = 'redcap_event_name'),\
                                                                     scid5_subst_med_bp, on = 'redcap_event_name'), \
                                                                     scid5_other_bp, on = 'redcap_event_name')
    scid5_all_bipolar_condition['value'] = np.where((scid5_all_bipolar_condition.filter(like='value').eq(1)).any(axis=1), 1,\
                                           np.where((scid5_all_bipolar_condition.filter(like='value').eq(0)).all(axis=1), 0, -900))
    scid5_all_bipolar = create_use_value('chrscid_any_bipolar', scid5_all_bipolar_condition, df_all, ['value'], voi_11, all_visits_list, 'int')
    scid5_all_depression_condition= pd.merge(pd.merge(pd.merge(pd.merge(scid5_mdd_wo_psychosis, scid5_persistent_depr, on = 'redcap_event_name'),\
                                                                     scid5_subst_med_depr, on = 'redcap_event_name'),\
                                                                     scid5_amc_depr, on = 'redcap_event_name'), \
                                                                     scid5_other_depr, on = 'redcap_event_name')
    scid5_all_depression_condition['value'] = np.where((scid5_all_depression_condition.filter(like='value').eq(1)).any(axis=1), 1,\
                                              np.where((scid5_all_depression_condition.filter(like='value').eq(0)).all(axis=1), 0, -900))
    scid5_all_depression = create_use_value('chrscid_any_depression', scid5_all_depression_condition, df_all, ['value'], voi_11, all_visits_list, 'int')
    scid5_all_mood_condition= pd.merge(scid5_all_depression, scid5_all_bipolar, on = 'redcap_event_name')
    scid5_all_mood_condition['value'] = np.where((scid5_all_mood_condition.filter(like='value').eq(1)).any(axis=1), 1,\
                                        np.where((scid5_all_mood_condition.filter(like='value').eq(0)).all(axis=1), 0, -900))
    scid5_all_mood = create_use_value('chrscid_any_mood', scid5_all_mood_condition, df_all, ['value'], voi_11, all_visits_list, 'int')
    # substance use disorder
    alcohol_disorder         = create_scid5_substance('chrscid_alcohol_use_disorder',       df_all, df_all, ['chrscid_e13_14', 'chrscid_e33'], voi_11, all_visits_list, 'int',        'chrscid_sedhypanx_yn') 
    sed_hyp_anx_disorder     = create_scid5_substance('chrscid_sedhypanx_use_disorder',     df_all, df_all, ['chrscid_e136_137', 'chrscid_e299_301'], voi_11, all_visits_list, 'int', 'chrscid_sedhypanx_yn')
    cannabis_disorder        = create_scid5_substance('chrscid_cannabis_use_disorder',      df_all, df_all, ['chrscid_e138_139', 'chrscid_e303_305'], voi_11, all_visits_list, 'int', 'chrscid_cannabis_yn')
    stimulants_disorder      = create_scid5_substance('chrscid_stimulants_use_disorder',    df_all, df_all, ['chrscid_e140_141', 'chrscid_e307_309'], voi_11, all_visits_list, 'int', 'chrscid_stimulant_yn')
    opiod_disorder           = create_scid5_substance('chrscid_opiod_use_disorder',         df_all, df_all, ['chrscid_e142_143', 'chrscid_e311_313'], voi_11, all_visits_list, 'int', 'chrscid_opioids_yn')
    inhalants_disorder       = create_scid5_substance('chrscid_inhalants_use_disorder',     df_all, df_all, ['chrscid_e144_145', 'chrscid_e315_317'], voi_11, all_visits_list, 'int', 'chrscid_inhalant_yn')
    pcp_disorder             = create_scid5_substance('chrscid_pcp_use_disorder',           df_all, df_all, ['chrscid_e146_147', 'chrscid_e319_321'], voi_11, all_visits_list, 'int', 'chrscid_sedhypanx_yn')
    hallucinogens_disorder   = create_scid5_substance('chrscid_halluc_use_disorder',        df_all, df_all, ['chrscid_e148_149', 'chrscid_e323_325'], voi_11, all_visits_list, 'int', 'chrscid_hallucinogen_yn')
    other_substance_disorder = create_scid5_substance('chrscid_other_use_disorder',         df_all, df_all, ['chrscid_e150_151', 'chrscid_e327_329'], voi_11, all_visits_list, 'int', 'chrscid_sedhypanx_yn')
    scid5_all_substance_condition= pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(alcohol_disorder, sed_hyp_anx_disorder, on = 'redcap_event_name'),\
                                                                                                           cannabis_disorder, on = 'redcap_event_name'),\
                                                                                                           stimulants_disorder, on = 'redcap_event_name'), \
                                                                                                           opiod_disorder, on = 'redcap_event_name'), \
                                                                                                           inhalants_disorder, on = 'redcap_event_name'),\
                                                                                                           pcp_disorder, on = 'redcap_event_name'), \
                                                                                                           hallucinogens_disorder, on = 'redcap_event_name'), \
                                                                                                           other_substance_disorder, on = 'redcap_event_name')
    scid5_all_substance_condition['value'] = np.where((scid5_all_substance_condition.filter(like='value').gt(0)).any(axis=1), scid5_all_substance_condition.filter(like='value').max(axis=1),\
                                             np.where((scid5_all_substance_condition.filter(like='value').eq(0)).all(axis=1), 0, -900))
    scid5_all_substance = create_use_value('chrscid_any_subst_use_disorder', scid5_all_substance_condition, df_all, ['value'], voi_11, all_visits_list, 'int')
    scid_all = pd.concat([scid5_delusional, scid5_brief_psychotic, scid5_schizophreniform, scid5_schizophrenia, scid5_schizoaffective_bipolar, scid5_schizoaffective_depression,\
                          scid5_schizoaffective_unspecified, scid5_subst_med_psychosis, scid5_amc_psychosis, scid5_other_psychosis,\
                          scid5_bp_wo_psychosis, scid5_bp_w_psychosis, scid5_bp_2, scid5_cyclothymic, scid5_subst_med_bp, scid5_other_bp,\
                          scid5_mdd_wo_psychosis, scid5_mdd_w_psychosis, scid5_persistent_depr, scid5_subst_med_depr, scid5_amc_depr, scid5_other_depr,\
                          alcohol_disorder, sed_hyp_anx_disorder, cannabis_disorder, stimulants_disorder, opiod_disorder, inhalants_disorder, pcp_disorder, hallucinogens_disorder,\
                          other_substance_disorder,\
                          scid5_all_psychosis, scid5_all_bipolar, scid5_all_depression, scid5_all_mood, scid5_all_substance], axis = 0)
    scid_all['data_type']='Integer'
    scid_all['ID']=id
    scid_all=scid_all[['variable','redcap_event_name','value','data_type','ID']]
# --------------------------------------------------------------------#
# PSYCHS-screening
# --------------------------------------------------------------------#
    # psychs
    psychs_pos_tot_scr = create_total_division('psychs_pos_tot', df_all, df_all, ['chrpsychs_scr_1d1','chrpsychs_scr_2d1','chrpsychs_scr_3d1','chrpsychs_scr_4d1',\
                                                                                  'chrpsychs_scr_5d1','chrpsychs_scr_6d1','chrpsychs_scr_7d1','chrpsychs_scr_8d1',\
                                                                                  'chrpsychs_scr_9d1','chrpsychs_scr_10d1','chrpsychs_scr_11d1','chrpsychs_scr_12d1',\
                                                                                  'chrpsychs_scr_13d1','chrpsychs_scr_14d1','chrpsychs_scr_15d1'], 1, voi_6, all_visits_list, 'int')
    # sips
    psychs_sips_p1_scr = create_max('psychs_sips_p1', df_all, df_all, ['chrpsychs_scr_1d1','chrpsychs_scr_3d1','chrpsychs_scr_4d1',\
                                                                       'chrpsychs_scr_5d1','chrpsychs_scr_6d1'], voi_6, all_visits_list, 'int')
    psychs_sips_p2_scr = create_use_value('psychs_sips_p2', df_all, df_all, ['chrpsychs_scr_2d1'], voi_6, all_visits_list, 'int')
    psychs_sips_p3_scr = create_max('psychs_sips_p3', df_all, df_all, ['chrpsychs_scr_7d1', 'chrpsychs_scr_8d1'], voi_6, all_visits_list, 'int')
    psychs_sips_p4_scr = create_max('psychs_sips_p4', df_all, df_all, ['chrpsychs_scr_9d1', 'chrpsychs_scr_10d1', 'chrpsychs_scr_11d1', 'chrpsychs_scr_12d1', 'chrpsychs_scr_13d1', 'chrpsychs_scr_14d1'],\
                                     voi_6, all_visits_list, 'int')
    psychs_sips_p5_scr = create_use_value('psychs_sips_p5', df_all, df_all, ['chrpsychs_scr_15d1'], voi_6, all_visits_list, 'int')
    sips_p1 = psychs_sips_p1_scr.copy()
    sips_p1['sips_p1'] = sips_p1['value']
    sips_p2 = psychs_sips_p2_scr.copy()
    sips_p2['sips_p2'] = sips_p2['value']
    sips_p3 = psychs_sips_p3_scr.copy()
    sips_p3['sips_p3'] = sips_p3['value']
    sips_p4 = psychs_sips_p4_scr.copy()
    sips_p4['sips_p4'] = sips_p4['value']
    sips_p5 = psychs_sips_p5_scr.copy()
    sips_p5['sips_p5'] = sips_p5['value']
    sips_scr = pd.merge(pd.merge(pd.merge(pd.merge(sips_p1, sips_p2, on = 'redcap_event_name'), sips_p3, on = 'redcap_event_name'), sips_p4, on = 'redcap_event_name'), sips_p5, on = 'redcap_event_name')
    sips_pos_tot_scr = create_total_division('sips_pos_tot', sips_scr, df_all, ['sips_p1', 'sips_p2', 'sips_p3', 'sips_p4', 'sips_p5'], 1, voi_6, all_visits_list, 'int')
    # caarms
    psychs_caarms_p1_scr = create_mul('psychs_caarms_p1', df_all, df_all, ['chrpsychs_scr_1d1','chrpsychs_scr_1d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p22_scr = create_mul('psychs_caarms_p22', df_all, df_all, ['chrpsychs_scr_2d1','chrpsychs_scr_2d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p22_scr['p22'] = psychs_caarms_p22_scr['value']
    psychs_caarms_p23_scr = create_mul('psychs_caarms_p23', df_all, df_all, ['chrpsychs_scr_3d1','chrpsychs_scr_3d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p23_scr['p23'] = psychs_caarms_p23_scr['value']
    psychs_caarms_p24_scr = create_mul('psychs_caarms_p24', df_all, df_all, ['chrpsychs_scr_4d1','chrpsychs_scr_4d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p24_scr['p24'] = psychs_caarms_p24_scr['value']
    psychs_caarms_p25_scr = create_mul('psychs_caarms_p25', df_all, df_all, ['chrpsychs_scr_5d1','chrpsychs_scr_5d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p25_scr['p25'] = psychs_caarms_p25_scr['value']
    psychs_caarms_p26_scr = create_mul('psychs_caarms_p26', df_all, df_all, ['chrpsychs_scr_6d1','chrpsychs_scr_6d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p26_scr['p26'] = psychs_caarms_p26_scr['value']
    psychs_caarms_p27_scr = create_mul('psychs_caarms_p27', df_all, df_all, ['chrpsychs_scr_7d1','chrpsychs_scr_7d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p27_scr['p27'] = psychs_caarms_p27_scr['value']
    psychs_caarms_p28_scr = create_mul('psychs_caarms_p28', df_all, df_all, ['chrpsychs_scr_8d1','chrpsychs_scr_8d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p28_scr['p28'] = psychs_caarms_p28_scr['value']
    psychs_caarms_p2 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(psychs_caarms_p22_scr, psychs_caarms_p23_scr, on = 'redcap_event_name'),\
                     psychs_caarms_p24_scr, on = 'redcap_event_name'), psychs_caarms_p25_scr, on = 'redcap_event_name'),\
                     psychs_caarms_p26_scr, on = 'redcap_event_name'), psychs_caarms_p27_scr, on = 'redcap_event_name'), psychs_caarms_p28_scr, on = 'redcap_event_name')
    psychs_caarms_p2_scr = create_max('psychs_caarms_p2', psychs_caarms_p2, df_all, ['p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28'], voi_6, all_visits_list, 'int')
    psychs_caarms_p32_scr = create_mul('psychs_caarms_p32', df_all, df_all, ['chrpsychs_scr_9d1','chrpsychs_scr_9d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p32_scr['p32'] = psychs_caarms_p32_scr['value']
    psychs_caarms_p33_scr = create_mul('psychs_caarms_p33', df_all, df_all, ['chrpsychs_scr_10d1','chrpsychs_scr_10d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p33_scr['p33'] = psychs_caarms_p33_scr['value']
    psychs_caarms_p34_scr = create_mul('psychs_caarms_p34', df_all, df_all, ['chrpsychs_scr_11d1','chrpsychs_scr_11d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p34_scr['p34'] = psychs_caarms_p34_scr['value']
    psychs_caarms_p35_scr = create_mul('psychs_caarms_p35', df_all, df_all, ['chrpsychs_scr_12d1','chrpsychs_scr_12d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p35_scr['p35'] = psychs_caarms_p35_scr['value']
    psychs_caarms_p36_scr = create_mul('psychs_caarms_p36', df_all, df_all, ['chrpsychs_scr_13d1','chrpsychs_scr_13d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p36_scr['p36'] = psychs_caarms_p36_scr['value']
    psychs_caarms_p37_scr = create_mul('psychs_caarms_p37', df_all, df_all, ['chrpsychs_scr_14d1','chrpsychs_scr_14d2'], voi_6, all_visits_list, 'int')
    psychs_caarms_p37_scr['p37'] = psychs_caarms_p37_scr['value']
    psychs_caarms_p3 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(psychs_caarms_p32_scr, psychs_caarms_p33_scr, on = 'redcap_event_name'),\
                     psychs_caarms_p34_scr, on = 'redcap_event_name'), psychs_caarms_p35_scr, on = 'redcap_event_name'),\
                     psychs_caarms_p36_scr, on = 'redcap_event_name'), psychs_caarms_p37_scr, on = 'redcap_event_name')
    psychs_caarms_p3_scr = create_max('psychs_caarms_p3', psychs_caarms_p3, df_all, ['p32', 'p33', 'p34', 'p35', 'p36', 'p37'], voi_6, all_visits_list, 'int')
    psychs_caarms_p4_scr = create_mul('psychs_caarms_p4', df_all, df_all, ['chrpsychs_scr_15d1','chrpsychs_scr_15d2'], voi_6, all_visits_list, 'int')
    caarms_p1 = psychs_caarms_p1_scr.copy()
    caarms_p1['caarms_p1'] = caarms_p1['value']
    caarms_p2 = psychs_caarms_p2_scr.copy()
    caarms_p2['caarms_p2'] = caarms_p2['value']
    caarms_p3 = psychs_caarms_p3_scr.copy()
    caarms_p3['caarms_p3'] = caarms_p3['value']
    caarms_p4 = psychs_caarms_p4_scr.copy()
    caarms_p4['caarms_p4'] = caarms_p4['value']
    caarms_scr = pd.merge(pd.merge(pd.merge(caarms_p1, caarms_p2, on = 'redcap_event_name'), caarms_p3, on = 'redcap_event_name'), caarms_p4, on = 'redcap_event_name')
    caarms_pos_tot_scr = create_total_division('caarms_pos_tot', caarms_scr, df_all, ['caarms_p1', 'caarms_p2', 'caarms_p3', 'caarms_p4'], 1, voi_6, all_visits_list, 'int')
    psychosis_onset_date_scr1 = create_min_date('psychosis_onset_date', df_all, df_all, ['chrpsychs_scr_1a6_on','chrpsychs_scr_2a6_on','chrpsychs_scr_3a6_on','chrpsychs_scr_4a6_on',\
                                                                                        'chrpsychs_scr_5a6_on','chrpsychs_scr_6a6_on','chrpsychs_scr_7a6_on','chrpsychs_scr_8a6_on',\
                                                                                        'chrpsychs_scr_9a6_on','chrpsychs_scr_10a6_on','chrpsychs_scr_11a6_on','chrpsychs_scr_12a6_on',\
                                                                                        'chrpsychs_scr_13a6_on','chrpsychs_scr_14a6_on','chrpsychs_scr_15a6_on'], voi_6, all_visits_list, 'str')
    psychosis_onset_relevant = df_all.copy()
    psychosis_onset_relevant['psychosis_onset_yes'] = psychosis_onset_relevant[['chrpsychs_scr_1a6','chrpsychs_scr_2a6','chrpsychs_scr_3a6','chrpsychs_scr_4a6',\
                                                                                'chrpsychs_scr_5a6','chrpsychs_scr_6a6','chrpsychs_scr_7a6','chrpsychs_scr_8a6',\
                                                                                'chrpsychs_scr_9a6','chrpsychs_scr_10a6','chrpsychs_scr_11a6','chrpsychs_scr_12a6',\
                                                                                'chrpsychs_scr_13a6','chrpsychs_scr_14a6','chrpsychs_scr_15a6']].isin(['1']).any(axis=1)
    psychosis_onset_relevant = psychosis_onset_relevant[['psychosis_onset_yes', 'redcap_event_name']]
    psychosis_onset_date_scr = pd.merge(psychosis_onset_date_scr1, psychosis_onset_relevant, on = 'redcap_event_name', how = 'left')
    psychosis_onset_date_scr['value'] = np.where(psychosis_onset_date_scr['psychosis_onset_yes'] == True, psychosis_onset_date_scr['value'],'1903-03-03')
    psychosis_onset_date_scr = psychosis_onset_date_scr[['variable', 'redcap_event_name', 'value']]
    psychosis_onset_date_use_calc = psychosis_onset_date_scr.copy()
    # we need to different date formats for the subsequent calculations (psychosis_onset_date_use_calc for calculating and psychosis_onset_date_scr to write out results in NDA format)
    psychosis_onset_date_use_calc['value'] = pd.to_datetime(psychosis_onset_date_use_calc['value'], format='%Y-%m-%d').dt.date
    psychosis_onset_date_scr['value'] = pd.to_datetime(psychosis_onset_date_scr['value'], format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
    psychosis_scr                            = create_use_value('chrpsychs_scr_ac1', df_all, df_all, ['chrpsychs_scr_ac1'], voi_6, all_visits_list, 'int')
    caarms_blips_scr                         = create_use_value('chrpsychs_scr_ac2', df_all, df_all, ['chrpsychs_scr_ac2'], voi_6, all_visits_list, 'int')
    caarms_aps_subthreshold_frequency_scr    = create_use_value('chrpsychs_scr_ac3', df_all, df_all, ['chrpsychs_scr_ac3'], voi_6, all_visits_list, 'int')
    caarms_aps_subthreshold_intensity_scr    = create_use_value('chrpsychs_scr_ac4', df_all, df_all, ['chrpsychs_scr_ac4'], voi_6, all_visits_list, 'int')
    caarms_aps_scr                           = create_use_value('chrpsychs_scr_ac5', df_all, df_all, ['chrpsychs_scr_ac5'], voi_6, all_visits_list, 'int')
    caarms_vulnerability_scr                 = create_use_value('chrpsychs_scr_ac6', df_all, df_all, ['chrpsychs_scr_ac6'], voi_6, all_visits_list, 'int')
    caarms_uhr_scr                           = create_use_value('chrpsychs_scr_ac7', df_all, df_all, ['chrpsychs_scr_ac7'], voi_6, all_visits_list, 'int')
    sips_bips_progression_scr                = create_use_value('chrpsychs_scr_ac9', df_all, df_all, ['chrpsychs_scr_ac9'], voi_6, all_visits_list, 'int')
    sips_bips_persistence_scr                = create_use_value('chrpsychs_scr_ac10', df_all, df_all, ['chrpsychs_scr_ac10'], voi_6, all_visits_list, 'int')
    sips_bips_partial_remission_scr          = create_use_value('chrpsychs_scr_ac11', df_all, df_all, ['chrpsychs_scr_ac11'], voi_6, all_visits_list, 'int')
    sips_bips_full_remission_scr             = create_use_value('chrpsychs_scr_ac12', df_all, df_all, ['chrpsychs_scr_ac12'], voi_6, all_visits_list, 'int')
    sips_apss_progression_scr                = create_use_value('chrpsychs_scr_ac15', df_all, df_all, ['chrpsychs_scr_ac15'], voi_6, all_visits_list, 'int')
    sips_apss_persistence_scr                = create_use_value('chrpsychs_scr_ac16', df_all, df_all, ['chrpsychs_scr_ac16'], voi_6, all_visits_list, 'int')
    sips_apss_partial_remission_scr          = create_use_value('chrpsychs_scr_ac17', df_all, df_all, ['chrpsychs_scr_ac17'], voi_6, all_visits_list, 'int')
    sips_apss_full_remission_scr             = create_use_value('chrpsychs_scr_ac18', df_all, df_all, ['chrpsychs_scr_ac18'], voi_6, all_visits_list, 'int')
    sips_grd_progression_scr                 = create_use_value('chrpsychs_scr_ac21', df_all, df_all, ['chrpsychs_scr_ac21'], voi_6, all_visits_list, 'int')
    sips_grd_persistence_scr                 = create_use_value('chrpsychs_scr_ac22', df_all, df_all, ['chrpsychs_scr_ac22'], voi_6, all_visits_list, 'int')
    sips_grd_partial_remission_scr           = create_use_value('chrpsychs_scr_ac23', df_all, df_all, ['chrpsychs_scr_ac23'], voi_6, all_visits_list, 'int')
    sips_grd_full_remission_scr              = create_use_value('chrpsychs_scr_ac24', df_all, df_all, ['chrpsychs_scr_ac24'], voi_6, all_visits_list, 'int')
    sips_chr_progression_scr                 = create_use_value('chrpsychs_scr_ac27', df_all, df_all, ['chrpsychs_scr_ac27'], voi_6, all_visits_list, 'int')
    sips_chr_persistence_scr                 = create_use_value('chrpsychs_scr_ac28', df_all, df_all, ['chrpsychs_scr_ac28'], voi_6, all_visits_list, 'int')
    sips_chr_partial_remission_scr           = create_use_value('chrpsychs_scr_ac29', df_all, df_all, ['chrpsychs_scr_ac29'], voi_6, all_visits_list, 'int')
    sips_chr_full_remission_scr              = create_use_value('chrpsychs_scr_ac30', df_all, df_all, ['chrpsychs_scr_ac30'], voi_6, all_visits_list, 'int')
    sips_current_status_scr                  = create_use_value('chrpsychs_scr_ac31', df_all, df_all, ['chrpsychs_scr_ac31'], voi_6, all_visits_list, 'int')
    dsm5_attenuated_psychosis_scr            = create_use_value('chrpsychs_scr_ac32', df_all, df_all, ['chrpsychs_scr_ac32'], voi_6, all_visits_list, 'int')
    # create the SIPS BIPS diagnosis at SCREENING
    bips_onsetdate_groups_scr =['chrpsychs_scr_1a10_on', 'chrpsychs_scr_2a10_on', 'chrpsychs_scr_3a10_on', 'chrpsychs_scr_4a10_on',\
                                'chrpsychs_scr_5a10_on', 'chrpsychs_scr_6a10_on', 'chrpsychs_scr_7a10_on', 'chrpsychs_scr_8a10_on',\
                                'chrpsychs_scr_9a10_on', 'chrpsychs_scr_10a10_on','chrpsychs_scr_11a10_on','chrpsychs_scr_12a10_on',\
                                'chrpsychs_scr_13a10_on','chrpsychs_scr_14a10_on','chrpsychs_scr_15a10_on']
    vars_interest_bips_scr = ['chrpsychs_scr_1a10', 'chrpsychs_scr_2a10', 'chrpsychs_scr_3a10', 'chrpsychs_scr_4a10', 'chrpsychs_scr_5a10',\
                              'chrpsychs_scr_6a10', 'chrpsychs_scr_7a10', 'chrpsychs_scr_8a10', 'chrpsychs_scr_9a10', 'chrpsychs_scr_10a10',\
                              'chrpsychs_scr_11a10','chrpsychs_scr_12a10','chrpsychs_scr_13a10','chrpsychs_scr_14a10','chrpsychs_scr_15a10']
    bips_new_final_scr = create_sips_groups_scr('sips_bips_scr_lifetime', df_all, bips_onsetdate_groups_scr, voi_6,\
                                                all_visits_list, psychosis_onset_date_use_calc, vars_interest_bips_scr, 'chrpsychs_scr_ac1')
    # create the SIPS APS diagnosis at SCREENING
    aps_onsetdate_groups_scr =['chrpsychs_scr_1a14_on', 'chrpsychs_scr_2a14_on', 'chrpsychs_scr_3a14_on', 'chrpsychs_scr_4a14_on',\
                                'chrpsychs_scr_5a14_on', 'chrpsychs_scr_6a14_on', 'chrpsychs_scr_7a14_on', 'chrpsychs_scr_8a14_on',\
                                'chrpsychs_scr_9a14_on', 'chrpsychs_scr_10a14_on','chrpsychs_scr_11a14_on','chrpsychs_scr_12a14_on',\
                                'chrpsychs_scr_13a14_on','chrpsychs_scr_14a14_on','chrpsychs_scr_15a14_on']
    vars_interest_aps_scr = ['chrpsychs_scr_1a14', 'chrpsychs_scr_2a14', 'chrpsychs_scr_3a14', 'chrpsychs_scr_4a14', 'chrpsychs_scr_5a14',\
                              'chrpsychs_scr_6a14', 'chrpsychs_scr_7a14', 'chrpsychs_scr_8a14', 'chrpsychs_scr_9a14', 'chrpsychs_scr_10a14',\
                              'chrpsychs_scr_11a14','chrpsychs_scr_12a14','chrpsychs_scr_13a14','chrpsychs_scr_14a14','chrpsychs_scr_15a14']
    aps_new_final_scr = create_sips_groups_scr('sips_aps_scr_lifetime', df_all, aps_onsetdate_groups_scr, voi_6,\
                                                all_visits_list, psychosis_onset_date_use_calc, vars_interest_aps_scr, 'chrpsychs_scr_ac1')
    # create the SIPS GRD diagnosis at SCREENING
    grd_onsetdate_groups_scr =['chrpsychs_scr_e4_date']
    vars_interest_grd_scr = ['chrpsychs_scr_e4']
    grd_new_final_scr = create_sips_groups_scr('sips_grd_scr_lifetime', df_all, grd_onsetdate_groups_scr, voi_6,\
                                                all_visits_list, psychosis_onset_date_use_calc, vars_interest_grd_scr, 'chrpsychs_scr_ac1')
    # create the CHR diagnosis at Screening
    sips_scr_chr = pd.merge(pd.merge(bips_new_final_scr, aps_new_final_scr, on = 'redcap_event_name'), grd_new_final_scr, on = 'redcap_event_name')
    sips_scr_chr['value'] = np.where((sips_scr_chr[['value', 'value_x', 'value_y']]==1).any(axis=1), 1, \
                            np.where((sips_scr_chr[['value', 'value_x', 'value_y']]==0).all(axis=1), 0, -900))
    sips_scr_chr['variable'] = 'sips_chr_scr_lifetime'
    sips_scr_chr = sips_scr_chr[['variable', 'redcap_event_name', 'value']]
    sips_scr_chr['value'] = np.where(~sips_scr_chr['redcap_event_name'].str.contains('screening'), '-300', sips_scr_chr['value'])
    df_visit_sips = df_all.copy()
    df_visit_sips = df_visit_sips[['redcap_event_name']]
    if df_visit_sips['redcap_event_name'].str.contains('arm_1').any():
        sips_scr_chr['value'] = np.where(sips_scr_chr['redcap_event_name'].str.contains('arm_2'), -300, sips_scr_chr['value'])
    elif df_visit_sips['redcap_event_name'].str.contains('arm_2').any():
        sips_scr_chr['value'] = np.where(sips_scr_chr['redcap_event_name'].str.contains('arm_1'), -300, sips_scr_chr['value'])
    else:
        warnings[11] = f"not the righ arm: 1380"
    # create the sips/bips/grd diagnosis from baseline for follow-up
    psychs_scr = pd.concat([psychs_pos_tot_scr, psychs_sips_p1_scr, psychs_sips_p2_scr, psychs_sips_p3_scr, psychs_sips_p4_scr, psychs_sips_p5_scr, sips_pos_tot_scr, psychs_caarms_p1_scr,\
                            psychs_caarms_p2_scr, psychs_caarms_p3_scr, psychs_caarms_p4_scr, caarms_pos_tot_scr, psychosis_onset_date_scr, psychosis_scr,\
                            caarms_blips_scr, caarms_aps_subthreshold_frequency_scr, caarms_aps_subthreshold_intensity_scr, caarms_aps_scr, caarms_vulnerability_scr, caarms_uhr_scr, \
                            sips_bips_progression_scr, sips_bips_persistence_scr, sips_bips_partial_remission_scr, sips_bips_full_remission_scr, sips_apss_progression_scr, sips_apss_persistence_scr,\
                            sips_apss_partial_remission_scr, sips_apss_full_remission_scr, sips_grd_progression_scr, sips_grd_persistence_scr, sips_grd_partial_remission_scr, \
                            sips_grd_full_remission_scr, sips_chr_progression_scr, sips_chr_persistence_scr, sips_chr_partial_remission_scr, sips_chr_full_remission_scr, \
                            sips_current_status_scr, dsm5_attenuated_psychosis_scr, bips_new_final_scr, aps_new_final_scr, grd_new_final_scr, sips_scr_chr], axis = 0) 
# --------------------------------------------------------------------#
# PSYCHS-follow-up
# --------------------------------------------------------------------#
    # In both networks HCs have the follow-up with hcpsychs and not with chrpsychs. Thus, we have to write this conditional
    if group == 'hc':
        # psychs
        psychs_pos_tot_fu = create_total_division('psychs_pos_tot', df_all, df_all, ['hcpsychs_fu_1d1','hcpsychs_fu_2d1','hcpsychs_fu_3d1','hcpsychs_fu_4d1',\
                                                                                      'hcpsychs_fu_5d1','hcpsychs_fu_6d1','hcpsychs_fu_7d1','hcpsychs_fu_8d1',\
                                                                                      'hcpsychs_fu_9d1','hcpsychs_fu_10d1','hcpsychs_fu_11d1','hcpsychs_fu_12d1',\
                                                                                      'hcpsychs_fu_13d1','hcpsychs_fu_14d1','hcpsychs_fu_15d1'], 1, voi_8, all_visits_list, 'int')
        # sips
        psychs_sips_p1_fu = create_max('psychs_sips_p1', df_all, df_all, ['hcpsychs_fu_1d1','hcpsychs_fu_3d1','hcpsychs_fu_4d1',\
                                                                           'hcpsychs_fu_5d1','hcpsychs_fu_6d1'], voi_8, all_visits_list, 'int')
        psychs_sips_p2_fu = create_use_value('psychs_sips_p2', df_all, df_all, ['hcpsychs_fu_2d1'], voi_8, all_visits_list, 'int')
        psychs_sips_p3_fu = create_max('psychs_sips_p3', df_all, df_all, ['hcpsychs_fu_7d1', 'hcpsychs_fu_8d1'], voi_8, all_visits_list, 'int')
        psychs_sips_p4_fu = create_max('psychs_sips_p4', df_all, df_all, ['hcpsychs_fu_9d1', 'hcpsychs_fu_10d1', 'hcpsychs_fu_11d1', 'hcpsychs_fu_12d1', 'hcpsychs_fu_13d1', 'hcpsychs_fu_14d1'],\
                                         voi_8, all_visits_list, 'int')
        psychs_sips_p5_fu = create_use_value('psychs_sips_p5', df_all, df_all, ['hcpsychs_fu_15d1'], voi_8, all_visits_list, 'int')
        sips_p1 = psychs_sips_p1_fu.copy()
        sips_p1['sips_p1'] = sips_p1['value']
        sips_p2 = psychs_sips_p2_fu.copy()
        sips_p2['sips_p2'] = sips_p2['value']
        sips_p3 = psychs_sips_p3_fu.copy()
        sips_p3['sips_p3'] = sips_p3['value']
        sips_p4 = psychs_sips_p4_fu.copy()
        sips_p4['sips_p4'] = sips_p4['value']
        sips_p5 = psychs_sips_p5_fu.copy()
        sips_p5['sips_p5'] = sips_p5['value']
        sips_fu = pd.merge(pd.merge(pd.merge(pd.merge(sips_p1, sips_p2, on = 'redcap_event_name'), sips_p3, on = 'redcap_event_name'), sips_p4, on = 'redcap_event_name'), sips_p5, on = 'redcap_event_name')
        sips_pos_tot_fu = create_total_division('sips_pos_tot', sips_fu, df_all, ['sips_p1', 'sips_p2', 'sips_p3', 'sips_p4', 'sips_p5'], 1, voi_8, all_visits_list, 'int')
        # caarms
        psychs_caarms_p1_fu = create_mul('psychs_caarms_p1', df_all, df_all, ['hcpsychs_fu_1d1','hcpsychs_fu_1d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p22_fu = create_mul('psychs_caarms_p22', df_all, df_all, ['hcpsychs_fu_2d1','hcpsychs_fu_2d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p22_fu['p22'] = psychs_caarms_p22_fu['value']
        psychs_caarms_p23_fu = create_mul('psychs_caarms_p23', df_all, df_all, ['hcpsychs_fu_3d1','hcpsychs_fu_3d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p23_fu['p23'] = psychs_caarms_p23_fu['value']
        psychs_caarms_p24_fu = create_mul('psychs_caarms_p24', df_all, df_all, ['hcpsychs_fu_4d1','hcpsychs_fu_4d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p24_fu['p24'] = psychs_caarms_p24_fu['value']
        psychs_caarms_p25_fu = create_mul('psychs_caarms_p25', df_all, df_all, ['hcpsychs_fu_5d1','hcpsychs_fu_5d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p25_fu['p25'] = psychs_caarms_p25_fu['value']
        psychs_caarms_p26_fu = create_mul('psychs_caarms_p26', df_all, df_all, ['hcpsychs_fu_6d1','hcpsychs_fu_6d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p26_fu['p26'] = psychs_caarms_p26_fu['value']
        psychs_caarms_p27_fu = create_mul('psychs_caarms_p27', df_all, df_all, ['hcpsychs_fu_7d1','hcpsychs_fu_7d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p27_fu['p27'] = psychs_caarms_p27_fu['value']
        psychs_caarms_p28_fu = create_mul('psychs_caarms_p28', df_all, df_all, ['hcpsychs_fu_8d1','hcpsychs_fu_8d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p28_fu['p28'] = psychs_caarms_p28_fu['value']
        psychs_caarms_p2 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(psychs_caarms_p22_fu, psychs_caarms_p23_fu, on = 'redcap_event_name'),\
                         psychs_caarms_p24_fu, on = 'redcap_event_name'), psychs_caarms_p25_fu, on = 'redcap_event_name'),\
                         psychs_caarms_p26_fu, on = 'redcap_event_name'), psychs_caarms_p27_fu, on = 'redcap_event_name'), psychs_caarms_p28_fu, on = 'redcap_event_name')
        psychs_caarms_p2_fu = create_max('psychs_caarms_p2', psychs_caarms_p2, df_all, ['p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28'], voi_8, all_visits_list, 'int')
        psychs_caarms_p32_fu = create_mul('psychs_caarms_p32', df_all, df_all, ['hcpsychs_fu_9d1','hcpsychs_fu_9d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p32_fu['p32'] = psychs_caarms_p32_fu['value']
        psychs_caarms_p33_fu = create_mul('psychs_caarms_p33', df_all, df_all, ['hcpsychs_fu_10d1','hcpsychs_fu_10d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p33_fu['p33'] = psychs_caarms_p33_fu['value']
        psychs_caarms_p34_fu = create_mul('psychs_caarms_p34', df_all, df_all, ['hcpsychs_fu_11d1','hcpsychs_fu_11d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p34_fu['p34'] = psychs_caarms_p34_fu['value']
        psychs_caarms_p35_fu = create_mul('psychs_caarms_p35', df_all, df_all, ['hcpsychs_fu_12d1','hcpsychs_fu_12d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p35_fu['p35'] = psychs_caarms_p35_fu['value']
        psychs_caarms_p36_fu = create_mul('psychs_caarms_p36', df_all, df_all, ['hcpsychs_fu_13d1','hcpsychs_fu_13d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p36_fu['p36'] = psychs_caarms_p36_fu['value']
        psychs_caarms_p37_fu = create_mul('psychs_caarms_p37', df_all, df_all, ['hcpsychs_fu_14d1','hcpsychs_fu_14d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p37_fu['p37'] = psychs_caarms_p37_fu['value']
        psychs_caarms_p3 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(psychs_caarms_p32_fu, psychs_caarms_p33_fu, on = 'redcap_event_name'),\
                         psychs_caarms_p34_fu, on = 'redcap_event_name'), psychs_caarms_p35_fu, on = 'redcap_event_name'),\
                         psychs_caarms_p36_fu, on = 'redcap_event_name'), psychs_caarms_p37_fu, on = 'redcap_event_name')
        psychs_caarms_p3_fu = create_max('psychs_caarms_p3', psychs_caarms_p3, df_all, ['p32', 'p33', 'p34', 'p35', 'p36', 'p37'], voi_8, all_visits_list, 'int')
        psychs_caarms_p4_fu = create_mul('psychs_caarms_p4', df_all, df_all, ['hcpsychs_fu_15d1','hcpsychs_fu_15d2'], voi_8, all_visits_list, 'int')
        caarms_p1 = psychs_caarms_p1_fu.copy()
        caarms_p1['caarms_p1'] = caarms_p1['value']
        caarms_p2 = psychs_caarms_p2_fu.copy()
        caarms_p2['caarms_p2'] = caarms_p2['value']
        caarms_p3 = psychs_caarms_p3_fu.copy()
        caarms_p3['caarms_p3'] = caarms_p3['value']
        caarms_p4 = psychs_caarms_p4_fu.copy()
        caarms_p4['caarms_p4'] = caarms_p4['value']
        caarms_fu = pd.merge(pd.merge(pd.merge(caarms_p1, caarms_p2, on = 'redcap_event_name'), caarms_p3, on = 'redcap_event_name'), caarms_p4, on = 'redcap_event_name')
        caarms_pos_tot_fu = create_total_division('caarms_pos_tot', caarms_fu, df_all, ['caarms_p1', 'caarms_p2', 'caarms_p3', 'caarms_p4'], 1, voi_8, all_visits_list, 'int')
        conversion_date_fu1 = create_min_date('conversion_date', df_all, df_all, ['hcpsychs_fu_1c6_on','hcpsychs_fu_2c6_on','hcpsychs_fu_3c6_on','hcpsychs_fu_4c6_on',\
                                                                                  'hcpsychs_fu_5c6_on','hcpsychs_fu_6c6_on','hcpsychs_fu_7c6_on','hcpsychs_fu_8c6_on',\
                                                                                  'hcpsychs_fu_9c6_on','hcpsychs_fu_10c6_on','hcpsychs_fu_11c6_on','hcpsychs_fu_12c6_on',\
                                                                                  'hcpsychs_fu_13c6_on','hcpsychs_fu_14c6_on','hcpsychs_fu_15c6_on'], voi_8, all_visits_list, 'str')
        conversion_date_fu_relevant = df_all.copy()
        conversion_date_fu_relevant['psychosis_conversion_yes'] = conversion_date_fu_relevant[['hcpsychs_fu_1c6','hcpsychs_fu_2c6','hcpsychs_fu_3c6','hcpsychs_fu_4c6',\
                                                                                               'hcpsychs_fu_5c6','hcpsychs_fu_6c6','hcpsychs_fu_7c6','hcpsychs_fu_8c6',\
                                                                                               'hcpsychs_fu_9c6','hcpsychs_fu_10c6','hcpsychs_fu_11c6','hcpsychs_fu_12c6',\
                                                                                               'hcpsychs_fu_13c6','hcpsychs_fu_14c6','hcpsychs_fu_15c6']].isin(['1']).any(axis=1)
        conversion_date_fu_relevant = conversion_date_fu_relevant[['psychosis_conversion_yes', 'redcap_event_name']]
        conversion_date_fu = pd.merge(conversion_date_fu1, conversion_date_fu_relevant, on='redcap_event_name', how = 'left')
        conversion_date_fu['value'] = np.where(conversion_date_fu['psychosis_conversion_yes'] == True, conversion_date_fu['value'], '1903-03-03')
        conversion_date_fu = conversion_date_fu[['variable', 'redcap_event_name', 'value']]
        conversion_date_use_calc = conversion_date_fu.copy()
        # we need to different date formats for the subsequent calculations (conversion_onset_date_use_calc for calculating and conversion_onset_date_fu to write out results in NDA format)
        conversion_date_use_calc['value'] = pd.to_datetime(conversion_date_use_calc['value'], format='%Y-%m-%d').dt.date
        conversion_date_fu['value'] = pd.to_datetime(conversion_date_fu['value'], format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
        psychosis_curr_fu_hc                       = create_use_value('hcpsychs_fu_ac1_curr', df_all, df_all, ['hcpsychs_fu_ac1_curr'], voi_8, all_visits_list, 'int')
        psychosis_prev_fu_hc                       = create_use_value('hcpsychs_fu_ac1_prev', df_all, df_all, ['hcpsychs_fu_ac1_prev'], voi_8, all_visits_list, 'int')
        psychosis_first_fu_hc                      = create_use_value('hcpsychs_fu_ac1_first', df_all, df_all, ['hcpsychs_fu_ac1_first'], voi_8, all_visits_list, 'int')
        psychosis_conv_fu_hc                       = create_use_value('hcpsychs_fu_ac1_conv', df_all, df_all, ['hcpsychs_fu_ac1_conv'], voi_8, all_visits_list, 'int')
        psychosis_fu_hc                            = create_use_value('hcpsychs_fu_ac1', df_all, df_all, ['hcpsychs_fu_ac1'], voi_8, all_visits_list, 'int')
        sips_bips_progression_fu_hc                = create_use_value('hcpsychs_fu_ac9', df_all, df_all, ['hcpsychs_fu_ac9'], voi_8, all_visits_list, 'int')
        sips_bips_persistence_fu_hc                = create_use_value('hcpsychs_fu_ac10', df_all, df_all, ['hcpsychs_fu_ac10'], voi_8, all_visits_list, 'int')
        sips_bips_partial_remission_fu_hc          = create_use_value('hcpsychs_fu_ac11', df_all, df_all, ['hcpsychs_fu_ac11'], voi_8, all_visits_list, 'int')
        sips_bips_full_remission_fu_hc             = create_use_value('hcpsychs_fu_ac12', df_all, df_all, ['hcpsychs_fu_ac12'], voi_8, all_visits_list, 'int')
        sips_apss_progression_fu_hc                = create_use_value('hcpsychs_fu_ac15', df_all, df_all, ['hcpsychs_fu_ac15'], voi_8, all_visits_list, 'int')
        sips_apss_persistence_fu_hc                = create_use_value('hcpsychs_fu_ac16', df_all, df_all, ['hcpsychs_fu_ac16'], voi_8, all_visits_list, 'int')
        sips_apss_partial_remission_fu_hc          = create_use_value('hcpsychs_fu_ac17', df_all, df_all, ['hcpsychs_fu_ac17'], voi_8, all_visits_list, 'int')
        sips_apss_full_remission_fu_hc             = create_use_value('hcpsychs_fu_ac18', df_all, df_all, ['hcpsychs_fu_ac18'], voi_8, all_visits_list, 'int')
        sips_grd_progression_fu_hc                 = create_use_value('hcpsychs_fu_ac21', df_all, df_all, ['hcpsychs_fu_ac21'], voi_8, all_visits_list, 'int')
        sips_grd_persistence_fu_hc                 = create_use_value('hcpsychs_fu_ac22', df_all, df_all, ['hcpsychs_fu_ac22'], voi_8, all_visits_list, 'int')
        sips_grd_partial_remission_fu_hc           = create_use_value('hcpsychs_fu_ac23', df_all, df_all, ['hcpsychs_fu_ac23'], voi_8, all_visits_list, 'int')
        sips_grd_full_remission_fu_hc              = create_use_value('hcpsychs_fu_ac24', df_all, df_all, ['hcpsychs_fu_ac24'], voi_8, all_visits_list, 'int')
        sips_chr_progression_fu_hc                 = create_use_value('hcpsychs_fu_ac27', df_all, df_all, ['hcpsychs_fu_ac27'], voi_8, all_visits_list, 'int')
        sips_chr_persistence_fu_hc                 = create_use_value('hcpsychs_fu_ac28', df_all, df_all, ['hcpsychs_fu_ac28'], voi_8, all_visits_list, 'int')
        sips_chr_partial_remission_fu_hc           = create_use_value('hcpsychs_fu_ac29', df_all, df_all, ['hcpsychs_fu_ac29'], voi_8, all_visits_list, 'int')
        sips_chr_full_remission_fu_hc              = create_use_value('hcpsychs_fu_ac30', df_all, df_all, ['hcpsychs_fu_ac30'], voi_8, all_visits_list, 'int')
        sips_current_status_fu_hc                  = create_use_value('hcpsychs_fu_ac31', df_all, df_all, ['hcpsychs_fu_ac31'], voi_8, all_visits_list, 'int')
        dsm5_attenuated_psychosis_fu_hc            = create_use_value('hcpsychs_fu_ac32', df_all, df_all, ['hcpsychs_fu_ac32'], voi_8, all_visits_list, 'int')
        psychosis_curr_fu_chr                      = create_use_value('chrpsychs_fu_ac1_curr', df_all, df_all, ['chrpsychs_fu_ac1_curr'], voi_8, all_visits_list, 'int')
        psychosis_prev_fu_chr                      = create_use_value('chrpsychs_fu_ac1_prev', df_all, df_all, ['chrpsychs_fu_ac1_prev'], voi_8, all_visits_list, 'int')
        psychosis_first_fu_chr                     = create_use_value('chrpsychs_fu_ac1_first', df_all, df_all,['chrpsychs_fu_ac1_first'], voi_8, all_visits_list, 'int')
        psychosis_conv_fu_chr                      = create_use_value('chrpsychs_fu_ac1_conv', df_all, df_all, ['chrpsychs_fu_ac1_conv'], voi_8, all_visits_list, 'int')
        psychosis_fu_chr                           = create_use_value('chrpsychs_fu_ac1', df_all, df_all,      ['chrpsychs_fu_ac1'], voi_8, all_visits_list, 'int')
        sips_bips_progression_fu_chr               = create_use_value('chrpsychs_fu_ac9', df_all, df_all,      ['chrpsychs_fu_ac9'], voi_8, all_visits_list, 'int')
        sips_bips_persistence_fu_chr               = create_use_value('chrpsychs_fu_ac10', df_all, df_all,     ['chrpsychs_fu_ac10'], voi_8, all_visits_list, 'int')
        sips_bips_partial_remission_fu_chr         = create_use_value('chrpsychs_fu_ac11', df_all, df_all,     ['chrpsychs_fu_ac11'], voi_8, all_visits_list, 'int')
        sips_bips_full_remission_fu_chr            = create_use_value('chrpsychs_fu_ac12', df_all, df_all,     ['chrpsychs_fu_ac12'], voi_8, all_visits_list, 'int')
        sips_apss_progression_fu_chr               = create_use_value('chrpsychs_fu_ac15', df_all, df_all,     ['chrpsychs_fu_ac15'], voi_8, all_visits_list, 'int')
        sips_apss_persistence_fu_chr               = create_use_value('chrpsychs_fu_ac16', df_all, df_all,     ['chrpsychs_fu_ac16'], voi_8, all_visits_list, 'int')
        sips_apss_partial_remission_fu_chr         = create_use_value('chrpsychs_fu_ac17', df_all, df_all,     ['chrpsychs_fu_ac17'], voi_8, all_visits_list, 'int')
        sips_apss_full_remission_fu_chr            = create_use_value('chrpsychs_fu_ac18', df_all, df_all,     ['chrpsychs_fu_ac18'], voi_8, all_visits_list, 'int')
        sips_grd_progression_fu_chr                = create_use_value('chrpsychs_fu_ac21', df_all, df_all,     ['chrpsychs_fu_ac21'], voi_8, all_visits_list, 'int')
        sips_grd_persistence_fu_chr                = create_use_value('chrpsychs_fu_ac22', df_all, df_all,     ['chrpsychs_fu_ac22'], voi_8, all_visits_list, 'int')
        sips_grd_partial_remission_fu_chr          = create_use_value('chrpsychs_fu_ac23', df_all, df_all,     ['chrpsychs_fu_ac23'], voi_8, all_visits_list, 'int')
        sips_grd_full_remission_fu_chr             = create_use_value('chrpsychs_fu_ac24', df_all, df_all,     ['chrpsychs_fu_ac24'], voi_8, all_visits_list, 'int')
        sips_chr_progression_fu_chr                = create_use_value('chrpsychs_fu_ac27', df_all, df_all,     ['chrpsychs_fu_ac27'], voi_8, all_visits_list, 'int')
        sips_chr_persistence_fu_chr                = create_use_value('chrpsychs_fu_ac28', df_all, df_all,     ['chrpsychs_fu_ac28'], voi_8, all_visits_list, 'int')
        sips_chr_partial_remission_fu_chr          = create_use_value('chrpsychs_fu_ac29', df_all, df_all,     ['chrpsychs_fu_ac29'], voi_8, all_visits_list, 'int')
        sips_chr_full_remission_fu_chr             = create_use_value('chrpsychs_fu_ac30', df_all, df_all,     ['chrpsychs_fu_ac30'], voi_8, all_visits_list, 'int')
        sips_current_status_fu_chr                 = create_use_value('chrpsychs_fu_ac31', df_all, df_all,     ['chrpsychs_fu_ac31'], voi_8, all_visits_list, 'int')
        dsm5_attenuated_psychosis_fu_chr           = create_use_value('chrpsychs_fu_ac32', df_all, df_all,     ['chrpsychs_fu_ac32'], voi_8, all_visits_list, 'int')
        # create the SIPS BIPS diagnosis
        scr_bips_vars = ['chrpsychs_scr_ac9', 'chrpsychs_scr_ac10','chrpsychs_scr_ac11', 'chrpsychs_scr_ac12']
        bips_onsetdate_groups =['hcpsychs_fu_1c10_on','hcpsychs_fu_2c10_on','hcpsychs_fu_3c10_on','hcpsychs_fu_4c10_on',\
                                'hcpsychs_fu_5c10_on','hcpsychs_fu_6c10_on','hcpsychs_fu_7c10_on','hcpsychs_fu_8c10_on',\
                                'hcpsychs_fu_9c10_on','hcpsychs_fu_10c10_on','hcpsychs_fu_11c10_on','hcpsychs_fu_12c10_on',\
                                'hcpsychs_fu_13c10_on','hcpsychs_fu_14c10_on','hcpsychs_fu_15c10_on']
        vars_interest_bips_fu = ['hcpsychs_fu_1c10','hcpsychs_fu_2c10','hcpsychs_fu_3c10','hcpsychs_fu_4c10','hcpsychs_fu_5c10',\
                                 'hcpsychs_fu_6c10','hcpsychs_fu_7c10','hcpsychs_fu_8c10','hcpsychs_fu_9c10','hcpsychs_fu_10c10',\
                                 'hcpsychs_fu_11c10','hcpsychs_fu_12c10','hcpsychs_fu_13c10','hcpsychs_fu_14c10','hcpsychs_fu_15c10']
        bips_new_final, bips_ac_final= create_sips_groups('sips_bips_fu_new', 'sips_bips_lifetime', df_all, bips_new_final_scr, bips_onsetdate_groups, voi_8,\
                                                          all_visits_list, conversion_date_use_calc, vars_interest_bips_fu, 'hcpsychs_fu_ac1_conv', voi_10)
        # create the SIPS APS diagnosis
        scr_aps_vars = ['chrpsychs_scr_ac15', 'chrpsychs_scr_ac16','chrpsychs_scr_ac17', 'chrpsychs_scr_ac18']
        aps_onsetdate_groups =['hcpsychs_fu_1c14_on','hcpsychs_fu_2c14_on','hcpsychs_fu_3c14_on','hcpsychs_fu_4c14_on',\
                               'hcpsychs_fu_5c14_on','hcpsychs_fu_6c14_on','hcpsychs_fu_7c14_on','hcpsychs_fu_8c14_on',\
                               'hcpsychs_fu_9c14_on','hcpsychs_fu_10c14_on','hcpsychs_fu_11c14_on','hcpsychs_fu_12c14_on',\
                               'hcpsychs_fu_13c14_on','hcpsychs_fu_14c14_on','hcpsychs_fu_15c14_on']
        vars_interest_aps_fu = ['hcpsychs_fu_1c14','hcpsychs_fu_2c14','hcpsychs_fu_3c14','hcpsychs_fu_4c14','hcpsychs_fu_5c14',\
                                 'hcpsychs_fu_6c14','hcpsychs_fu_7c14','hcpsychs_fu_8c14','hcpsychs_fu_9c14','hcpsychs_fu_10c14',\
                                 'hcpsychs_fu_11c14','hcpsychs_fu_12c14','hcpsychs_fu_13c14','hcpsychs_fu_14c14','hcpsychs_fu_15c14']
        aps_new_final, aps_ac_final= create_sips_groups('sips_aps_fu_new', 'sips_aps_lifetime', df_all, aps_new_final_scr, aps_onsetdate_groups, voi_8,\
                                                        all_visits_list, conversion_date_use_calc, vars_interest_aps_fu, 'hcpsychs_fu_ac1_conv', voi_10)
        # create the SIPS GRD diagnosis the calculation is a little bit different!
        scr_grd_vars = ['chrpsychs_scr_ac21', 'chrpsychs_scr_ac22','chrpsychs_scr_ac23', 'chrpsychs_scr_ac24']
        grd_onsetdate_groups =['hcpsychs_fu_e4_date']
        vars_interest_grd_fu = ['hcpsychs_fu_e4_new']
        grd_new_final, grd_ac_final= create_sips_groups('sips_grd_fu_new', 'sips_grd_lifetime', df_all, grd_new_final_scr, grd_onsetdate_groups, voi_8,\
                                                        all_visits_list, conversion_date_use_calc, vars_interest_grd_fu, 'hcpsychs_fu_ac1_conv', voi_10)
        # create the CHR diagnosis at follow-up 
        sips_fu_chr = pd.merge(pd.merge(bips_ac_final, aps_ac_final, on = 'redcap_event_name'), grd_ac_final, on = 'redcap_event_name')
        sips_fu_chr['value'] = np.where((sips_fu_chr[['value', 'value_x', 'value_y']]==1).any(axis=1), 1, \
                                np.where((sips_fu_chr[['value', 'value_x', 'value_y']]==0).all(axis=1), 0, -900))
        sips_fu_chr['variable'] = 'sips_chr_lifetime'
        sips_fu_chr = sips_fu_chr[['variable', 'redcap_event_name', 'value']]
        sips_fu_chr['value'] = np.where(~sips_fu_chr['redcap_event_name'].str.contains(voi_10), '-300', sips_fu_chr['value'])
        df_visit_sips = df_all.copy()
        df_visit_sips = df_visit_sips[['redcap_event_name']]
        if df_visit_sips['redcap_event_name'].str.contains('arm_1').any():
            sips_fu_chr['value'] = np.where(sips_fu_chr['redcap_event_name'].str.contains('arm_2'), -300, sips_fu_chr['value'])
        elif df_visit_sips['redcap_event_name'].str.contains('arm_2').any():
            sips_fu_chr['value'] = np.where(sips_fu_chr['redcap_event_name'].str.contains('arm_1'), -300, sips_fu_chr['value'])
        # Combine the psychs fu dataframes
        psychs_fu = pd.concat([psychs_pos_tot_fu,psychs_sips_p1_fu,psychs_sips_p2_fu,psychs_sips_p3_fu,psychs_sips_p4_fu,psychs_sips_p5_fu,\
                               sips_pos_tot_fu, psychs_caarms_p1_fu,\
                               psychs_caarms_p2_fu, psychs_caarms_p3_fu, psychs_caarms_p4_fu, caarms_pos_tot_fu, conversion_date_fu, psychosis_fu_chr, psychosis_fu_hc,\
                               psychosis_curr_fu_hc, psychosis_prev_fu_hc, psychosis_first_fu_hc, psychosis_conv_fu_hc,\
                               sips_bips_progression_fu_hc, sips_bips_persistence_fu_hc, sips_bips_partial_remission_fu_hc, sips_bips_full_remission_fu_hc, 
                               sips_apss_progression_fu_hc, sips_apss_persistence_fu_hc,\
                               sips_apss_partial_remission_fu_hc, sips_apss_full_remission_fu_hc, sips_grd_progression_fu_hc, sips_grd_persistence_fu_hc, sips_grd_partial_remission_fu_hc, \
                               sips_grd_full_remission_fu_hc, sips_chr_progression_fu_hc, sips_chr_persistence_fu_hc, sips_chr_partial_remission_fu_hc, sips_chr_full_remission_fu_hc, \
                               sips_current_status_fu_hc, dsm5_attenuated_psychosis_fu_hc,\
                               psychosis_curr_fu_chr, psychosis_prev_fu_chr, psychosis_first_fu_chr, psychosis_conv_fu_chr,\
                               sips_bips_progression_fu_chr, sips_bips_persistence_fu_chr, sips_bips_partial_remission_fu_chr, sips_bips_full_remission_fu_chr, 
                               sips_apss_progression_fu_chr, sips_apss_persistence_fu_chr,\
                               sips_apss_partial_remission_fu_chr, sips_apss_full_remission_fu_chr, sips_grd_progression_fu_chr, sips_grd_persistence_fu_chr, sips_grd_partial_remission_fu_chr, \
                               sips_grd_full_remission_fu_chr, sips_chr_progression_fu_chr, sips_chr_persistence_fu_chr, sips_chr_partial_remission_fu_chr, sips_chr_full_remission_fu_chr, \
                               sips_current_status_fu_chr, dsm5_attenuated_psychosis_fu_chr, bips_new_final, bips_ac_final, aps_new_final, aps_ac_final, grd_new_final, grd_ac_final,\
                               sips_fu_chr], axis = 0) 
    else:
        # psychs
        psychs_pos_tot_fu = create_total_division('psychs_pos_tot', df_all, df_all, ['chrpsychs_fu_1d1','chrpsychs_fu_2d1','chrpsychs_fu_3d1','chrpsychs_fu_4d1',\
                                                                                      'chrpsychs_fu_5d1','chrpsychs_fu_6d1','chrpsychs_fu_7d1','chrpsychs_fu_8d1',\
                                                                                      'chrpsychs_fu_9d1','chrpsychs_fu_10d1','chrpsychs_fu_11d1','chrpsychs_fu_12d1',\
                                                                                      'chrpsychs_fu_13d1','chrpsychs_fu_14d1','chrpsychs_fu_15d1'], 1, voi_8, all_visits_list, 'int')
        # sips
        psychs_sips_p1_fu = create_max('psychs_sips_p1', df_all, df_all, ['chrpsychs_fu_1d1','chrpsychs_fu_3d1','chrpsychs_fu_4d1',\
                                                                           'chrpsychs_fu_5d1','chrpsychs_fu_6d1'], voi_8, all_visits_list, 'int')
        psychs_sips_p2_fu = create_use_value('psychs_sips_p2', df_all, df_all, ['chrpsychs_fu_2d1'], voi_8, all_visits_list, 'int')
        psychs_sips_p3_fu = create_max('psychs_sips_p3', df_all, df_all, ['chrpsychs_fu_7d1', 'chrpsychs_fu_8d1'], voi_8, all_visits_list, 'int')
        psychs_sips_p4_fu = create_max('psychs_sips_p4', df_all, df_all, ['chrpsychs_fu_9d1', 'chrpsychs_fu_10d1', 'chrpsychs_fu_11d1', 'chrpsychs_fu_12d1', 'chrpsychs_fu_13d1', 'chrpsychs_fu_14d1'],\
                                         voi_8, all_visits_list, 'int')
        psychs_sips_p5_fu = create_use_value('psychs_sips_p5', df_all, df_all, ['chrpsychs_fu_15d1'], voi_8, all_visits_list, 'int')
        sips_p1 = psychs_sips_p1_fu.copy()
        sips_p1['sips_p1'] = sips_p1['value']
        sips_p2 = psychs_sips_p2_fu.copy()
        sips_p2['sips_p2'] = sips_p2['value']
        sips_p3 = psychs_sips_p3_fu.copy()
        sips_p3['sips_p3'] = sips_p3['value']
        sips_p4 = psychs_sips_p4_fu.copy()
        sips_p4['sips_p4'] = sips_p4['value']
        sips_p5 = psychs_sips_p5_fu.copy()
        sips_p5['sips_p5'] = sips_p5['value']
        sips_fu = pd.merge(pd.merge(pd.merge(pd.merge(sips_p1, sips_p2, on = 'redcap_event_name'), sips_p3, on = 'redcap_event_name'), sips_p4, on = 'redcap_event_name'), sips_p5, on = 'redcap_event_name')
        sips_pos_tot_fu = create_total_division('sips_pos_tot', sips_fu, df_all, ['sips_p1', 'sips_p2', 'sips_p3', 'sips_p4', 'sips_p5'], 1, voi_8, all_visits_list, 'int')
        # caarms
        psychs_caarms_p1_fu = create_mul('psychs_caarms_p1', df_all, df_all, ['chrpsychs_fu_1d1','chrpsychs_fu_1d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p22_fu = create_mul('psychs_caarms_p22', df_all, df_all, ['chrpsychs_fu_2d1','chrpsychs_fu_2d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p22_fu['p22'] = psychs_caarms_p22_fu['value']
        psychs_caarms_p23_fu = create_mul('psychs_caarms_p23', df_all, df_all, ['chrpsychs_fu_3d1','chrpsychs_fu_3d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p23_fu['p23'] = psychs_caarms_p23_fu['value']
        psychs_caarms_p24_fu = create_mul('psychs_caarms_p24', df_all, df_all, ['chrpsychs_fu_4d1','chrpsychs_fu_4d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p24_fu['p24'] = psychs_caarms_p24_fu['value']
        psychs_caarms_p25_fu = create_mul('psychs_caarms_p25', df_all, df_all, ['chrpsychs_fu_5d1','chrpsychs_fu_5d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p25_fu['p25'] = psychs_caarms_p25_fu['value']
        psychs_caarms_p26_fu = create_mul('psychs_caarms_p26', df_all, df_all, ['chrpsychs_fu_6d1','chrpsychs_fu_6d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p26_fu['p26'] = psychs_caarms_p26_fu['value']
        psychs_caarms_p27_fu = create_mul('psychs_caarms_p27', df_all, df_all, ['chrpsychs_fu_7d1','chrpsychs_fu_7d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p27_fu['p27'] = psychs_caarms_p27_fu['value']
        psychs_caarms_p28_fu = create_mul('psychs_caarms_p28', df_all, df_all, ['chrpsychs_fu_8d1','chrpsychs_fu_8d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p28_fu['p28'] = psychs_caarms_p28_fu['value']
        psychs_caarms_p2 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(psychs_caarms_p22_fu, psychs_caarms_p23_fu, on = 'redcap_event_name'),\
                         psychs_caarms_p24_fu, on = 'redcap_event_name'), psychs_caarms_p25_fu, on = 'redcap_event_name'),\
                         psychs_caarms_p26_fu, on = 'redcap_event_name'), psychs_caarms_p27_fu, on = 'redcap_event_name'), psychs_caarms_p28_fu, on = 'redcap_event_name')
        psychs_caarms_p2_fu = create_max('psychs_caarms_p2', psychs_caarms_p2, df_all, ['p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28'], voi_8, all_visits_list, 'int')
        psychs_caarms_p32_fu = create_mul('psychs_caarms_p32', df_all, df_all, ['chrpsychs_fu_9d1','chrpsychs_fu_9d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p32_fu['p32'] = psychs_caarms_p32_fu['value']
        psychs_caarms_p33_fu = create_mul('psychs_caarms_p33', df_all, df_all, ['chrpsychs_fu_10d1','chrpsychs_fu_10d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p33_fu['p33'] = psychs_caarms_p33_fu['value']
        psychs_caarms_p34_fu = create_mul('psychs_caarms_p34', df_all, df_all, ['chrpsychs_fu_11d1','chrpsychs_fu_11d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p34_fu['p34'] = psychs_caarms_p34_fu['value']
        psychs_caarms_p35_fu = create_mul('psychs_caarms_p35', df_all, df_all, ['chrpsychs_fu_12d1','chrpsychs_fu_12d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p35_fu['p35'] = psychs_caarms_p35_fu['value']
        psychs_caarms_p36_fu = create_mul('psychs_caarms_p36', df_all, df_all, ['chrpsychs_fu_13d1','chrpsychs_fu_13d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p36_fu['p36'] = psychs_caarms_p36_fu['value']
        psychs_caarms_p37_fu = create_mul('psychs_caarms_p37', df_all, df_all, ['chrpsychs_fu_14d1','chrpsychs_fu_14d2'], voi_8, all_visits_list, 'int')
        psychs_caarms_p37_fu['p37'] = psychs_caarms_p37_fu['value']
        psychs_caarms_p3 = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(psychs_caarms_p32_fu, psychs_caarms_p33_fu, on = 'redcap_event_name'),\
                         psychs_caarms_p34_fu, on = 'redcap_event_name'), psychs_caarms_p35_fu, on = 'redcap_event_name'),\
                         psychs_caarms_p36_fu, on = 'redcap_event_name'), psychs_caarms_p37_fu, on = 'redcap_event_name')
        psychs_caarms_p3_fu = create_max('psychs_caarms_p3', psychs_caarms_p3, df_all, ['p32', 'p33', 'p34', 'p35', 'p36', 'p37'], voi_8, all_visits_list, 'int')
        psychs_caarms_p4_fu = create_mul('psychs_caarms_p4', df_all, df_all, ['chrpsychs_fu_15d1','chrpsychs_fu_15d2'], voi_8, all_visits_list, 'int')
        caarms_p1 = psychs_caarms_p1_fu.copy()
        caarms_p1['caarms_p1'] = caarms_p1['value']
        caarms_p2 = psychs_caarms_p2_fu.copy()
        caarms_p2['caarms_p2'] = caarms_p2['value']
        caarms_p3 = psychs_caarms_p3_fu.copy()
        caarms_p3['caarms_p3'] = caarms_p3['value']
        caarms_p4 = psychs_caarms_p4_fu.copy()
        caarms_p4['caarms_p4'] = caarms_p4['value']
        caarms_fu = pd.merge(pd.merge(pd.merge(caarms_p1, caarms_p2, on = 'redcap_event_name'), caarms_p3, on = 'redcap_event_name'), caarms_p4, on = 'redcap_event_name')
        caarms_pos_tot_fu = create_total_division('caarms_pos_tot', caarms_fu, df_all, ['caarms_p1', 'caarms_p2', 'caarms_p3', 'caarms_p4'], 1, voi_8, all_visits_list, 'int')
        conversion_date_fu1 = create_min_date('conversion_date', df_all, df_all, ['chrpsychs_fu_1c6_on','chrpsychs_fu_2c6_on','chrpsychs_fu_3c6_on','chrpsychs_fu_4c6_on',\
                                                                                  'chrpsychs_fu_5c6_on','chrpsychs_fu_6c6_on','chrpsychs_fu_7c6_on','chrpsychs_fu_8c6_on',\
                                                                                  'chrpsychs_fu_9c6_on','chrpsychs_fu_10c6_on','chrpsychs_fu_11c6_on','chrpsychs_fu_12c6_on',\
                                                                                  'chrpsychs_fu_13c6_on','chrpsychs_fu_14c6_on','chrpsychs_fu_15c6_on'], voi_8, all_visits_list, 'str')
        conversion_date_fu_relevant = df_all.copy()
        conversion_date_fu_relevant['psychosis_conversion_yes'] = conversion_date_fu_relevant[['chrpsychs_fu_1c6','chrpsychs_fu_2c6','chrpsychs_fu_3c6','chrpsychs_fu_4c6',\
                                                                                               'chrpsychs_fu_5c6','chrpsychs_fu_6c6','chrpsychs_fu_7c6','chrpsychs_fu_8c6',\
                                                                                               'chrpsychs_fu_9c6','chrpsychs_fu_10c6','chrpsychs_fu_11c6','chrpsychs_fu_12c6',\
                                                                                               'chrpsychs_fu_13c6','chrpsychs_fu_14c6','chrpsychs_fu_15c6']].isin(['1']).any(axis=1)
        conversion_date_fu_relevant = conversion_date_fu_relevant[['psychosis_conversion_yes', 'redcap_event_name']]
        conversion_date_fu = pd.merge(conversion_date_fu1, conversion_date_fu_relevant, on='redcap_event_name', how='left')
        conversion_date_fu['value'] = np.where(conversion_date_fu['psychosis_conversion_yes'] == True, conversion_date_fu['value'], '1903-03-03')
        conversion_date_fu = conversion_date_fu[['variable', 'redcap_event_name', 'value']]
        conversion_date_use_calc = conversion_date_fu.copy()
        # we need to different date formats for the subsequent calculations (conversion_onset_date_use_calc for calculating and conversion_onset_date_fu to write out results in NDA format)
        conversion_date_use_calc['value'] = pd.to_datetime(conversion_date_use_calc['value'], format='%Y-%m-%d').dt.date
        conversion_date_fu['value'] = pd.to_datetime(conversion_date_fu['value'], format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
        psychosis_curr_fu_hc                       = create_use_value('hcpsychs_fu_ac1_curr', df_all, df_all, ['hcpsychs_fu_ac1_curr'], voi_8, all_visits_list, 'int')
        psychosis_prev_fu_hc                       = create_use_value('hcpsychs_fu_ac1_prev', df_all, df_all, ['hcpsychs_fu_ac1_prev'], voi_8, all_visits_list, 'int')
        psychosis_first_fu_hc                      = create_use_value('hcpsychs_fu_ac1_first', df_all, df_all, ['hcpsychs_fu_ac1_first'], voi_8, all_visits_list, 'int')
        psychosis_conv_fu_hc                       = create_use_value('hcpsychs_fu_ac1_conv', df_all, df_all, ['hcpsychs_fu_ac1_conv'], voi_8, all_visits_list, 'int')
        psychosis_fu_hc                            = create_use_value('hcpsychs_fu_ac1', df_all, df_all, ['hcpsychs_fu_ac1'], voi_8, all_visits_list, 'int')
        sips_bips_progression_fu_hc                = create_use_value('hcpsychs_fu_ac9', df_all, df_all, ['hcpsychs_fu_ac9'], voi_8, all_visits_list, 'int')
        sips_bips_persistence_fu_hc                = create_use_value('hcpsychs_fu_ac10', df_all, df_all, ['hcpsychs_fu_ac10'], voi_8, all_visits_list, 'int')
        sips_bips_partial_remission_fu_hc          = create_use_value('hcpsychs_fu_ac11', df_all, df_all, ['hcpsychs_fu_ac11'], voi_8, all_visits_list, 'int')
        sips_bips_full_remission_fu_hc             = create_use_value('hcpsychs_fu_ac12', df_all, df_all, ['hcpsychs_fu_ac12'], voi_8, all_visits_list, 'int')
        sips_apss_progression_fu_hc                = create_use_value('hcpsychs_fu_ac15', df_all, df_all, ['hcpsychs_fu_ac15'], voi_8, all_visits_list, 'int')
        sips_apss_persistence_fu_hc                = create_use_value('hcpsychs_fu_ac16', df_all, df_all, ['hcpsychs_fu_ac16'], voi_8, all_visits_list, 'int')
        sips_apss_partial_remission_fu_hc          = create_use_value('hcpsychs_fu_ac17', df_all, df_all, ['hcpsychs_fu_ac17'], voi_8, all_visits_list, 'int')
        sips_apss_full_remission_fu_hc             = create_use_value('hcpsychs_fu_ac18', df_all, df_all, ['hcpsychs_fu_ac18'], voi_8, all_visits_list, 'int')
        sips_grd_progression_fu_hc                 = create_use_value('hcpsychs_fu_ac21', df_all, df_all, ['hcpsychs_fu_ac21'], voi_8, all_visits_list, 'int')
        sips_grd_persistence_fu_hc                 = create_use_value('hcpsychs_fu_ac22', df_all, df_all, ['hcpsychs_fu_ac22'], voi_8, all_visits_list, 'int')
        sips_grd_partial_remission_fu_hc           = create_use_value('hcpsychs_fu_ac23', df_all, df_all, ['hcpsychs_fu_ac23'], voi_8, all_visits_list, 'int')
        sips_grd_full_remission_fu_hc              = create_use_value('hcpsychs_fu_ac24', df_all, df_all, ['hcpsychs_fu_ac24'], voi_8, all_visits_list, 'int')
        sips_chr_progression_fu_hc                 = create_use_value('hcpsychs_fu_ac27', df_all, df_all, ['hcpsychs_fu_ac27'], voi_8, all_visits_list, 'int')
        sips_chr_persistence_fu_hc                 = create_use_value('hcpsychs_fu_ac28', df_all, df_all, ['hcpsychs_fu_ac28'], voi_8, all_visits_list, 'int')
        sips_chr_partial_remission_fu_hc           = create_use_value('hcpsychs_fu_ac29', df_all, df_all, ['hcpsychs_fu_ac29'], voi_8, all_visits_list, 'int')
        sips_chr_full_remission_fu_hc              = create_use_value('hcpsychs_fu_ac30', df_all, df_all, ['hcpsychs_fu_ac30'], voi_8, all_visits_list, 'int')
        sips_current_status_fu_hc                  = create_use_value('hcpsychs_fu_ac31', df_all, df_all, ['hcpsychs_fu_ac31'], voi_8, all_visits_list, 'int')
        dsm5_attenuated_psychosis_fu_hc            = create_use_value('hcpsychs_fu_ac32', df_all, df_all, ['hcpsychs_fu_ac32'], voi_8, all_visits_list, 'int')
        psychosis_curr_fu_chr                      = create_use_value('chrpsychs_fu_ac1_curr', df_all, df_all, ['chrpsychs_fu_ac1_curr'], voi_8, all_visits_list, 'int')
        psychosis_prev_fu_chr                      = create_use_value('chrpsychs_fu_ac1_prev', df_all, df_all, ['chrpsychs_fu_ac1_prev'], voi_8, all_visits_list, 'int')
        psychosis_first_fu_chr                     = create_use_value('chrpsychs_fu_ac1_first', df_all, df_all, ['chrpsychs_fu_ac1_first'], voi_8, all_visits_list, 'int')
        psychosis_conv_fu_chr                      = create_use_value('chrpsychs_fu_ac1_conv', df_all, df_all, ['chrpsychs_fu_ac1_conv'], voi_8, all_visits_list, 'int')
        psychosis_fu_chr                           = create_use_value('chrpsychs_fu_ac1', df_all, df_all, ['chrpsychs_fu_ac1'], voi_8, all_visits_list, 'int')
        sips_bips_progression_fu_chr               = create_use_value('chrpsychs_fu_ac9', df_all, df_all, ['chrpsychs_fu_ac9'], voi_8, all_visits_list, 'int')
        sips_bips_persistence_fu_chr               = create_use_value('chrpsychs_fu_ac10', df_all, df_all, ['chrpsychs_fu_ac10'], voi_8, all_visits_list, 'int')
        sips_bips_partial_remission_fu_chr         = create_use_value('chrpsychs_fu_ac11', df_all, df_all, ['chrpsychs_fu_ac11'], voi_8, all_visits_list, 'int')
        sips_bips_full_remission_fu_chr            = create_use_value('chrpsychs_fu_ac12', df_all, df_all, ['chrpsychs_fu_ac12'], voi_8, all_visits_list, 'int')
        sips_apss_progression_fu_chr               = create_use_value('chrpsychs_fu_ac15', df_all, df_all, ['chrpsychs_fu_ac15'], voi_8, all_visits_list, 'int')
        sips_apss_persistence_fu_chr               = create_use_value('chrpsychs_fu_ac16', df_all, df_all, ['chrpsychs_fu_ac16'], voi_8, all_visits_list, 'int')
        sips_apss_partial_remission_fu_chr         = create_use_value('chrpsychs_fu_ac17', df_all, df_all, ['chrpsychs_fu_ac17'], voi_8, all_visits_list, 'int')
        sips_apss_full_remission_fu_chr            = create_use_value('chrpsychs_fu_ac18', df_all, df_all, ['chrpsychs_fu_ac18'], voi_8, all_visits_list, 'int')
        sips_grd_progression_fu_chr                = create_use_value('chrpsychs_fu_ac21', df_all, df_all, ['chrpsychs_fu_ac21'], voi_8, all_visits_list, 'int')
        sips_grd_persistence_fu_chr                = create_use_value('chrpsychs_fu_ac22', df_all, df_all, ['chrpsychs_fu_ac22'], voi_8, all_visits_list, 'int')
        sips_grd_partial_remission_fu_chr          = create_use_value('chrpsychs_fu_ac23', df_all, df_all, ['chrpsychs_fu_ac23'], voi_8, all_visits_list, 'int')
        sips_grd_full_remission_fu_chr             = create_use_value('chrpsychs_fu_ac24', df_all, df_all, ['chrpsychs_fu_ac24'], voi_8, all_visits_list, 'int')
        sips_chr_progression_fu_chr                = create_use_value('chrpsychs_fu_ac27', df_all, df_all, ['chrpsychs_fu_ac27'], voi_8, all_visits_list, 'int')
        sips_chr_persistence_fu_chr                = create_use_value('chrpsychs_fu_ac28', df_all, df_all, ['chrpsychs_fu_ac28'], voi_8, all_visits_list, 'int')
        sips_chr_partial_remission_fu_chr          = create_use_value('chrpsychs_fu_ac29', df_all, df_all, ['chrpsychs_fu_ac29'], voi_8, all_visits_list, 'int')
        sips_chr_full_remission_fu_chr             = create_use_value('chrpsychs_fu_ac30', df_all, df_all, ['chrpsychs_fu_ac30'], voi_8, all_visits_list, 'int')
        sips_current_status_fu_chr                 = create_use_value('chrpsychs_fu_ac31', df_all, df_all, ['chrpsychs_fu_ac31'], voi_8, all_visits_list, 'int')
        dsm5_attenuated_psychosis_fu_chr           = create_use_value('chrpsychs_fu_ac32', df_all, df_all, ['chrpsychs_fu_ac32'], voi_8, all_visits_list, 'int')
        # create the SIPS BIPS diagnosis
        bips_onsetdate_groups =['chrpsychs_fu_1c10_on','chrpsychs_fu_2c10_on','chrpsychs_fu_3c10_on','chrpsychs_fu_4c10_on',\
                                'chrpsychs_fu_5c10_on','chrpsychs_fu_6c10_on','chrpsychs_fu_7c10_on','chrpsychs_fu_8c10_on',\
                                'chrpsychs_fu_9c10_on','chrpsychs_fu_10c10_on','chrpsychs_fu_11c10_on','chrpsychs_fu_12c10_on',\
                                'chrpsychs_fu_13c10_on','chrpsychs_fu_14c10_on','chrpsychs_fu_15c10_on']
        vars_interest_bips_fu = ['chrpsychs_fu_1c10','chrpsychs_fu_2c10','chrpsychs_fu_3c10','chrpsychs_fu_4c10','chrpsychs_fu_5c10',\
                                 'chrpsychs_fu_6c10','chrpsychs_fu_7c10','chrpsychs_fu_8c10','chrpsychs_fu_9c10','chrpsychs_fu_10c10',\
                                 'chrpsychs_fu_11c10','chrpsychs_fu_12c10','chrpsychs_fu_13c10','chrpsychs_fu_14c10','chrpsychs_fu_15c10']
        bips_new_final, bips_ac_final= create_sips_groups('sips_bips_fu_new', 'sips_bips_lifetime', df_all, bips_new_final_scr, bips_onsetdate_groups, voi_8,\
                                                          all_visits_list, conversion_date_use_calc, vars_interest_bips_fu, 'chrpsychs_fu_ac1_conv', voi_10)
        # create the SIPS APS diagnosis
        aps_onsetdate_groups =['chrpsychs_fu_1c14_on','chrpsychs_fu_2c14_on','chrpsychs_fu_3c14_on','chrpsychs_fu_4c14_on',\
                               'chrpsychs_fu_5c14_on','chrpsychs_fu_6c14_on','chrpsychs_fu_7c14_on','chrpsychs_fu_8c14_on',\
                               'chrpsychs_fu_9c14_on','chrpsychs_fu_10c14_on','chrpsychs_fu_11c14_on','chrpsychs_fu_12c14_on',\
                               'chrpsychs_fu_13c14_on','chrpsychs_fu_14c14_on','chrpsychs_fu_15c14_on']
        vars_interest_aps_fu = ['chrpsychs_fu_1c14','chrpsychs_fu_2c14','chrpsychs_fu_3c14','chrpsychs_fu_4c14','chrpsychs_fu_5c14',\
                                 'chrpsychs_fu_6c14','chrpsychs_fu_7c14','chrpsychs_fu_8c14','chrpsychs_fu_9c14','chrpsychs_fu_10c14',\
                                 'chrpsychs_fu_11c14','chrpsychs_fu_12c14','chrpsychs_fu_13c14','chrpsychs_fu_14c14','chrpsychs_fu_15c14']
        aps_new_final, aps_ac_final= create_sips_groups('sips_aps_fu_new', 'sips_aps_lifetime', df_all, aps_new_final_scr, aps_onsetdate_groups, voi_8,\
                                                        all_visits_list, conversion_date_use_calc, vars_interest_aps_fu, 'chrpsychs_fu_ac1_conv', voi_10)
        # create the SIPS GRD diagnosis the calculation is a little bit different!
        grd_onsetdate_groups =['chrpsychs_fu_e4_date']
        vars_interest_grd_fu = ['chrpsychs_fu_e4_new']
        grd_new_final, grd_ac_final= create_sips_groups('sips_grd_fu_new', 'sips_grd_lifetime', df_all, grd_new_final_scr, grd_onsetdate_groups, voi_8,\
                                                        all_visits_list, conversion_date_use_calc, vars_interest_grd_fu, 'chrpsychs_fu_ac1_conv', voi_10)
        # create the CHR diagnosis at follow-up 
        sips_fu_chr = pd.merge(pd.merge(bips_ac_final, aps_ac_final, on = 'redcap_event_name'), grd_ac_final, on = 'redcap_event_name')
        sips_fu_chr['value'] = np.where((sips_fu_chr[['value', 'value_x', 'value_y']]==1).any(axis=1), 1, \
                                np.where((sips_fu_chr[['value', 'value_x', 'value_y']]==0).all(axis=1), 0, -900))
        sips_fu_chr['variable'] = 'sips_chr_lifetime'
        sips_fu_chr = sips_fu_chr[['variable', 'redcap_event_name', 'value']]
        sips_fu_chr['value'] = np.where(~sips_fu_chr['redcap_event_name'].str.contains(voi_10), '-300', sips_fu_chr['value'])
        df_visit_sips = df_all.copy()
        df_visit_sips = df_visit_sips[['redcap_event_name']]
        if df_visit_sips['redcap_event_name'].str.contains('arm_1').any():
            sips_fu_chr['value'] = np.where(sips_fu_chr['redcap_event_name'].str.contains('arm_2'), -300, sips_fu_chr['value'])
        elif df_visit_sips['redcap_event_name'].str.contains('arm_2').any():
            sips_fu_chr['value'] = np.where(sips_fu_chr['redcap_event_name'].str.contains('arm_1'), -300, sips_fu_chr['value'])
        # combine the psychs_fu dataframes
        psychs_fu = pd.concat([psychs_pos_tot_fu, psychs_sips_p1_fu, psychs_sips_p2_fu, psychs_sips_p3_fu, psychs_sips_p4_fu,\
                               psychs_sips_p5_fu, sips_pos_tot_fu, psychs_caarms_p1_fu,\
                               psychs_caarms_p2_fu, psychs_caarms_p3_fu, psychs_caarms_p4_fu, caarms_pos_tot_fu, conversion_date_fu, psychosis_fu_chr, psychosis_fu_hc,\
                               psychosis_curr_fu_hc, psychosis_prev_fu_hc, psychosis_first_fu_hc, psychosis_conv_fu_hc,\
                               sips_bips_progression_fu_hc, sips_bips_persistence_fu_hc, sips_bips_partial_remission_fu_hc, sips_bips_full_remission_fu_hc, 
                               sips_apss_progression_fu_hc, sips_apss_persistence_fu_hc,\
                               sips_apss_partial_remission_fu_hc, sips_apss_full_remission_fu_hc, sips_grd_progression_fu_hc, sips_grd_persistence_fu_hc, sips_grd_partial_remission_fu_hc, \
                               sips_grd_full_remission_fu_hc, sips_chr_progression_fu_hc, sips_chr_persistence_fu_hc, sips_chr_partial_remission_fu_hc, sips_chr_full_remission_fu_hc, \
                               sips_current_status_fu_hc, dsm5_attenuated_psychosis_fu_hc,\
                               psychosis_curr_fu_chr, psychosis_prev_fu_chr, psychosis_first_fu_chr, psychosis_conv_fu_chr,\
                               sips_bips_progression_fu_chr, sips_bips_persistence_fu_chr, sips_bips_partial_remission_fu_chr, sips_bips_full_remission_fu_chr, 
                               sips_apss_progression_fu_chr, sips_apss_persistence_fu_chr,\
                               sips_apss_partial_remission_fu_chr, sips_apss_full_remission_fu_chr, sips_grd_progression_fu_chr, sips_grd_persistence_fu_chr, sips_grd_partial_remission_fu_chr, \
                               sips_grd_full_remission_fu_chr, sips_chr_progression_fu_chr, sips_chr_persistence_fu_chr, sips_chr_partial_remission_fu_chr, sips_chr_full_remission_fu_chr, \
                               sips_current_status_fu_chr, dsm5_attenuated_psychosis_fu_chr, bips_new_final, bips_ac_final, aps_new_final, aps_ac_final, grd_new_final, grd_ac_final,\
                               sips_fu_chr], axis = 0) 
    psychs_fu['value_fu'] = psychs_fu['value']
    psychs_scr['value_scr'] = psychs_scr['value']
    # we have to combine the psychs and the psychs_fu
    psychs_merged = pd.merge(psychs_scr, psychs_fu, on = ['redcap_event_name', 'variable'], how = 'outer')
    psychs_merged['value_scr'] = psychs_merged['value_scr'].fillna('-300').astype(str)
    psychs_merged['value_fu'] = psychs_merged['value_fu'].fillna('-300').astype(str)
    psychs_merged['value'] = np.where(psychs_merged['value_scr'] == '-300', psychs_merged['value_fu'], psychs_merged['value_scr'])
    psychs = psychs_merged[['variable', 'redcap_event_name', 'value']]
    psychs['data_type'] = np.where((psychs['variable'] == 'psychosis_onset_date') | (psychs['variable'] == 'conversion_date'), 'Date', 'Integer')
    if version == 'run_outcome' or version == 'single_subject':
        psychs.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/psychs.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        cdss.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/cdss.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        pdt.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/perceived_discrimination_scale.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        oasis.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/oasis.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        pss.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/perceived_stress_scale.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        bprs.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/bprs.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        gfr.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/global_functioning_role_scale.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        gfs.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/global_functioning_social_scale.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        pds_final.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/pubertal_developmental_scale.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        sofas_screening.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/sofas_screening.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        sofas_fu.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/sofas_followup.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        nsipr.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/nsipr.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        promis.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/item_promis_for_sleep.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        pgi_s.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/pgis.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        ra.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/ra_prediction.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        cssrs.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/cssrs_baseline.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        cssrs_fu_both.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/cssrs_followup.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        premorbid_adjustment.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/premorbid_adjustment_scale.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        assist.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/assist.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        polyrisk.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/psychosis_polyrisk_score.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')
        scid_all.to_csv('/data/predict1/data_from_nda/{0}/PHOENIX/GENERAL/{0}{1}/processed/{2}/surveys/scid5_psychosis_mood_substance_abuse.csv'.format(Network, site, id), index = False, header=True, float_format='%.3f')

    output_df_id = pd.concat([psychs, polyrisk, assist, premorbid_adjustment,cssrs, cssrs_fu_both, ra, pgi_s, promis, gfr, gfs, nsipr, sofas_screening, sofas_fu, pds_final, bprs, oasis, pdt, cdss, pss, scid_all],\
                   axis = 0, sort = True)
    output_df_id['ID'] = id 

    #'''throughout the script I have collected warnings. These warnings are now summarized and will be printed to individual subjects'''
    concat_logs = '\n'.join([formatted_date_warn] + [str(warning) for warning in warnings])
    all_empty = all(not s.strip() for s in warnings)
    # create a new folder per date including the logs
    date_with_time = datetime.strptime(formatted_date_warn, '%Y-%m-%d %H:%M:%S')
    date_folder_warn = date_with_time.strftime('%Y-%m-%d')
    new_log_folder = f'/data/predict1/home/np487/amp_scz/logs/{date_folder_warn}_{network}_{version}'
    os.makedirs(new_log_folder, exist_ok = True)
    warning_file_path = f'{new_log_folder}/warnings_{id}.txt'
    existing_content = ''
    # read the file from before:
    if all_empty == False:
        try:
            with open(warning_file_path, 'r') as file:
                existing_content = file.read()
        except FileNotFoundError:
            pass # File doesn't exist yet, which is fine
        
        # Combine the new message (date) with the existing content:
        updated_warning = f'{concat_logs}\n{existing_content}'
        # Write the updated content back to the file
        with open(warning_file_path, "w") as file:
            file.write(updated_warning)
        print(f"Log written to {warning_file_path}")

    return output_df_id


# --------------------------------------------------------------------#
# Here we load the data
# --------------------------------------------------------------------#

all_visits_list = ['screening_arm_1', 'baseline_arm_1', 'floating_forms_arm_1', 'month_1_arm_1', 'month_2_arm_1', 'month_3_arm_1', 'month_4_arm_1', 'month_5_arm_1', 'month_6_arm_1',\
                   'month_7_arm_1', 'month_8_arm_1', 'month_9_arm_1', 'month_10_arm_1', 'month_11_arm_1', 'month_12_arm_1', 'month_18_arm_1','month_24_arm_1', 'conversion_arm_1',\
                   'screening_arm_2', 'baseline_arm_2', 'floating_forms_arm_2', 'month_1_arm_2', 'month_2_arm_2', 'month_3_arm_2', 'month_4_arm_2', 'month_5_arm_2', 'month_6_arm_2',\
                   'month_7_arm_2', 'month_8_arm_2', 'month_9_arm_2', 'month_10_arm_2', 'month_11_arm_2', 'month_12_arm_2', 'month_18_arm_2','month_24_arm_2', 'conversion_arm_2']

network = sys.argv[1]
Network = network.capitalize()
version = sys.argv[2]


# I create the list of ids within the folder: /data/predict1/home/np487/amp_scz/create_list/ using hte bash script create list
# ids = pd.read_csv('/data/pnl/home/gj936/U24/Clinical_qc/flowqc/REAL_DATA/{0}_sub_list.txt'.format(network), sep= '\n', index_col = False, header = None)
ids = pd.read_csv('/data/predict1/home/np487/amp_scz/create_list/{0}_sub_list.txt'.format(network), index_col = False, header = None)

# Load the data. Depending on which network you load the data from you have to apply some different wrangling.
if Network == 'Pronet':
    if version == 'test' or version == 'create_control':
        id_list = ['YA16606', 'YA01508', 'LA00145', 'LA00834', 'OR00697', 'PI01355', 'HA04408']
    elif version == 'single_subject':
        id_list = ['OR05031']
    elif version == 'run_outcome':
        id_list = ids.iloc[:, 0].tolist()
    
elif Network == 'Prescient':
    if version == 'test' or version == 'create_control':
        id_list = ['ME00772', 'ME78581','BM90491', 'ME33634', 'ME20845', 'BM73097', 'ME21922']
    elif version == 'single_subject':
        id_list = ['ME41251', 'ME36620', 'ME83306', 'ME41273', 'ME31119']
    elif version == 'run_outcome':
        id_list = ids.iloc[:, 0].tolist()

num_workers = multiprocessing.cpu_count() // 2
print("Number of processes: {0}".format(num_workers))
pool = multiprocessing.Pool(num_workers)
outcomes = pool.map(compute_outcomes, id_list)

pool.close()
pool.join()

subject_list = []
outcomes = [outcome for outcome in outcomes if outcome is not None]
concatenated_df = pd.concat(outcomes)

# I should calculate all the warning messages
#outcomes = [outcome for outcome in outcomes if outcome is not None]
#concatenated_df = pd.concat(outcomes)

if version == 'test':
    print("Wrote the test - subjects to the control_subjects folder.")
    concatenated_df.to_csv("/data/predict1/home/np487/amp_scz/test_subjects/{0}_new_subjects.csv".format(network), index = False, header=True, float_format='%.3f')
elif version == 'create_control':
    concatenated_df.to_csv("/data/predict1/home/np487/amp_scz/{0}_test_subjects.csv".format(network), index = False, header=True, float_format='%.3f')
elif version == 'run_outcome':
    print("Wrote the real outcome - subjects to the control_subjects folder.")
    concatenated_df.to_csv("/data/predict1/home/np487/control_subjects/{0}_all.csv".format(network), index = False, header=True, float_format='%.3f')
    # To know when the outcome_calculation script was run the last time for all subjects create a text file:
    current_date = datetime.now()
    # Format the date as a string
    formatted_date = current_date.strftime('%Y-%m-%d %H:%M:%S')
    # Create the message
    message = f'{Network}: outcome calculations were run for the last time on: {formatted_date}'
    file_path = '/data/predict1/home/np487/amp_scz/outcome_calculations/last_date_runoutcome.txt'
    # read the file from before:
    existing_content = ''
    try:
        with open(file_path, 'r') as file:
            existing_content = file.read()
    except FileNotFoundError:
        pass # File doesn't exist yet, which is fine
    
    # Combine the new message (date) with the existing content:
    updated_content = f'{message}\n{existing_content}'
    # Write the updated content back to the file
    with open(file_path, "w") as file:
        file.write(updated_content)
    print(f"Log written to {file_path}")
elif version == 'single_subject':
    # To know when the outcome_calculation script was run the last time for all subjects create a text file:
    current_date = datetime.now()
    # Format the date as a string
    formatted_date = current_date.strftime('%Y-%m-%d %H:%M:%S')
    # Create the message
    message = f'Single Subject {id_list} outcome for {Network} were run for the last time on: {formatted_date}'
    file_path = '/data/predict1/home/np487/amp_scz/outcome_calculations/last_date_runoutcome.txt'
    # read the file from before:
    existing_content = ''
    try:
        with open(file_path, 'r') as file:
            existing_content = file.read()
    except FileNotFoundError:
        pass # File doesn't exist yet, which is fine
    
    # Combine the new message (date) with the existing content:
    updated_content = f'{message}\n{existing_content}'
    # Write the updated content back to the file
    with open(file_path, "w") as file:
        file.write(updated_content)
    print(f"Log written to {file_path}")

