import numpy as np

# Creating functions for simple imputation techniques (Mean, Median, Mode and Group-Based)
def global_mean(x):
  return x.fillna(x.mean())

def global_median(x):
  return x.fillna(x.median())

def global_mode(x):
  return x.fillna(x.mode()[0])

def mean_grouping(df,group_col,target_col):
  return df[target_col].fillna(df.groupby(group_col)[target_col].transform('mean'))

def median_grouping(df,group_col,target_col):
  return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

def mode_grouping(df,group_col,target_col):
    def compute_mode(x):
        return x.mode()[0] if not x.mode().empty else np.nan
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform(lambda x: compute_mode(x)))
