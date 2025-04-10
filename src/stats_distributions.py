# Numerical Stats and Distributions Function

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import skew,kurtosis,shapiro,probplot,shapiro,spearmanr,mstats
from sklearn.linear_model import BayesianRidge
import statsmodels.api as sm
from scipy.stats import chi2_contingency,zscore
from sklearn.preprocessing import QuantileTransformer,RobustScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest

def statistical_analysis(x,column_name = 'Variable',data = None,n = 10):
  mean = np.nanmean(x) # Note that the nan is used to compute the statistics despite the variable having null values.
  median = np.nanmedian(x)
  min = np.nanmin(x)
  max = np.nanmax(x)
  variance = np.nanvar(x)
  std_dev = np.nanstd(x)
  data_type = x.dtype
  null_vals = x.isnull().sum()
  null_vals_prop = (null_vals/len(data))*100
  skewness = skew(x.dropna())
  kurtosis_val = kurtosis(x.dropna()) + 3 # Adjusting for Fishers Kurtosis
  stat, p = shapiro(x.dropna())
  print(f'Mean of {column_name}: {mean}')
  print(f'Median of {column_name}: {median}')
  print(f'Minimum of {column_name}: {min}')
  print(f'Maximum of {column_name}: {max}')
  print(f'Variance of {column_name}: {variance}')
  print(f'Std_Dev of {column_name}: {std_dev}')
  print(f'Data Type of {column_name}: {data_type}')
  print(f'Null Vals of {column_name}: {null_vals}')
  print(f'Prop of Null Vals (%) of {column_name}: {null_vals_prop}')
  print(f'Skewness of {column_name}: {skewness}')
  print(f'Kurtosis of {column_name}: {kurtosis_val}')
  print(f'Shapiro-Wilk Test for {column_name}:Statistics = {stat},p-value = {p}')

  # Histogram (Distribution)
  plt.hist(x.dropna(),bins = n,color = 'blue',edgecolor = 'black')
  plt.title(f'Histogram of {column_name}')
  plt.xlabel('Values')
  plt.ylabel('Frequency')
  plt.show()
  plt.clf()

  # Box Plot (Distribution)
  plt.figure(figsize=(10, 6))
  plt.boxplot(x.dropna(),labels = [column_name],notch = True,patch_artist = True,
    boxprops = dict(facecolor = 'lightblue',color = 'blue'),
    medianprops=dict(color = 'red'))
  plt.title(f'Box-Plot of {column_name}')
  plt.ylabel('Values')
  plt.show()
  plt.clf()

  # Q-Q Plot (Normality Check)
  fig, ax = plt.subplots()
  probplot(x.dropna(),dist= "norm",plot = ax)
  ax.get_lines()[1].set_color('r')  # Reference line --> Red
  ax.get_lines()[1].set_linestyle('--')  # Dashed line
  plt.title(f'Q-Q Plot of {column_name}')
  plt.show()
  plt.clf()