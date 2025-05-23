

import numpy as np # Importing NumPy for numerical operations and handling arrays.
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data visualization, builds on matplotlib with better visual defaults
import matplotlib.pyplot as plt #Importing Pyplot from matplotlib for creating static, interactive, and animated visualizations in Python
import xgboost as xgb # a library for efficient implementation of gradient boosting framework
from xgboost import plot_importance, plot_tree # plotting feature importance and decision trees.
from sklearn.metrics import mean_squared_error, mean_absolute_error #performance analysis metrics for regression analysis
from sklearn.metrics import r2_score #r^2 score
import pickle #serializing and deserializing python object structures
plt.style.use('fivethirtyeight') ## Setting the visual style of matplotlib plots to 'fivethirtyeight' 