import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def transform_mydata(data):
    ## To drop variables from a dataframe, simply list the ones you want to keep. Can pass to a new dataframe (as here) or drop from original
    df = data
    df = pd.DataFrame(df, columns=['YARDSTOGO', 'SCOREDIFF', 'TIMELEFT', 'DOWNNAME','TRUEFIELD','RUN_PASS',
       'CONCEPT_8', 'CONCEPT_9','YARDSGAINED_Mean', 'AVG_OVER_TOGO', 'prob_FIRST_YES','pred_FIRST'])

    ## Re-Name columns as needed by casting new column names using the .columns method. New columns names will be applied in order and must be same length as number of existing columns.
    df.columns = ['YARDSTOGO', 'SCOREDIFF', 'TIMELEFT', 'DOWNNAME','TRUEFIELD','RUN_PASS',
       'CONCEPT_8', 'CONCEPT_9','YARDSGAINED_Mean', 'AVG_OVER_TOGO', 'prob_FIRST','TARGET_1ST']

    ## Convert numeric fields to numeric so that they can be explored and used for analysis
    df[['YARDSTOGO','SCOREDIFF','TIMELEFT','TRUEFIELD','YARDSGAINED_Mean','AVG_OVER_TOGO',
       'CONCEPT_8','CONCEPT_9','prob_FIRST']] = df[['YARDSTOGO','SCOREDIFF','TIMELEFT','TRUEFIELD','YARDSGAINED_Mean','AVG_OVER_TOGO',
       'CONCEPT_8','CONCEPT_9','prob_FIRST']].apply(pd.to_numeric)

    df.TARGET_1ST = df.TARGET_1ST.replace({'N': 0,'Y': 1})

    ## Use the bins= parameter to specify the desired cut points for bins 
    df['TEE'] = pd.cut(df.YARDSTOGO, bins=[0, 3, 6, 10, 100],  labels=["SHORT","MED","LONG","XLONG"], include_lowest=True)

    ## For sample stratification, we need to add a variable to capture the fields on which we want to stratify
    df['STRAT_1ST'] = df['TEE'].astype(str) + df['DOWNNAME'].astype(str) + df['TARGET_1ST'].astype(str)
        
    df['PLAYID'] = df['RUN_PASS'] + df['CONCEPT_8'].astype(str)

    bin_data = pd.get_dummies(df[['RUN_PASS','PLAYID','TEE','DOWNNAME']]) ## list the fields which you want to encode
    bin_data = bin_data.drop(['RUN_PASS_PASS'], axis=1) ## where the target field is binary, first field can be dropped

    df = df.drop(['RUN_PASS','PLAYID','TEE','DOWNNAME'], axis=1) ## drop original fields
    df = pd.concat([df, bin_data], axis=1)
    
    return(df)