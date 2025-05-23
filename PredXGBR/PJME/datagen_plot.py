
from packages import * #import all modules and libraries

# Loading and preprocessing data
pjme = pd.read_csv('../../../PJM_Load/PJME_hourly.csv', index_col=[0], parse_dates=[0])



# Defining color palette for plots
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]

#plotting the entire dataset
ax1= pjme.plot(style='.', figsize=(15,5), color='green', fontsize=13)
ax1.set_facecolor('white')
plt.xlabel("Time Series", fontsize=16)
plt.ylabel("Electrical Load (W)", fontsize=16)
plt.title("PJME Load", fontsize=18)
plt.show()



# Defining the split date for training and testing datasets
split_date = '02-Jan-2015'
df_train = pjme.loc[pjme.index <= split_date].copy()
df_test = pjme.loc[pjme.index > split_date].copy()




# Plotting training and testing sets with different colors
ax2= df_test \
    .rename(columns={'Load': 'TEST SET'}) \
    .join(df_train.rename(columns={'Load': 'TRAINING SET'}), how='outer') \
    .plot(figsize=(15,5), color=['green', 'rosybrown'],fontsize=13, style='.')
ax2.set_facecolor('white')
plt.xlabel("Time Series", fontsize=16)
plt.ylabel("Electrical Load (W)", fontsize=16)
plt.title("PJME Load", fontsize=18)
plt.show()



# Previewing data manipulation methods
df_train.head(7) # Displays the first 7 rows of the training dataset
df_train.head(7).shift(1) # Shifts the first 7 rows down by 1.
df_train.head(7)
df_train.head(7).rolling(window = 2).mean() # Computes the 2-point moving average of the first 7 rows



def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['pjme_6_hrs_lag'] = df['Load'].shift(6)
    df['pjme_12_hrs_lag'] = df['Load'].shift(12)
    df['pjme_24_hrs_lag'] = df['Load'].shift(24)
    df['pjme_6_hrs_mean'] = df['Load'].rolling(window = 6).mean()
    df['pjme_12_hrs_mean'] = df['Load'].rolling(window = 12).mean()
    df['pjme_24_hrs_mean'] = df['Load'].rolling(window = 24).mean()
    df['pjme_6_hrs_std'] = df['Load'].rolling(window = 6).std()
    df['pjme_12_hrs_std'] = df['Load'].rolling(window = 12).std()
    df['pjme_24_hrs_std'] = df['Load'].rolling(window = 24).std()
    df['pjme_6_hrs_max'] = df['Load'].rolling(window = 6).max()
    df['pjme_12_hrs_max'] = df['Load'].rolling(window = 12).max()
    df['pjme_24_hrs_max'] = df['Load'].rolling(window = 24).max()
    df['pjme_6_hrs_min'] = df['Load'].rolling(window = 6).min()
    df['pjme_12_hrs_min'] = df['Load'].rolling(window = 12).min()
    df['pjme_24_hrs_min'] = df['Load'].rolling(window = 24).min()
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear' , 'pjme_6_hrs_lag' , 'pjme_24_hrs_lag' , 'pjme_6_hrs_mean',
           "pjme_12_hrs_mean" ,"pjme_24_hrs_mean" ,"pjme_6_hrs_std" ,"pjme_12_hrs_std" ,"pjme_24_hrs_std",
           "pjme_6_hrs_max","pjme_12_hrs_max" ,"pjme_24_hrs_max" ,"pjme_6_hrs_min","pjme_12_hrs_min" ,"pjme_24_hrs_min"]]
    if label:
        y = df[label] #Extracting the label for supervised learning
        return X, y
    return X

# Creating training and testing feature matrices and label vectors
X_train, y_train = create_features(df_train, label='Load')
X_test, y_test = create_features(df_test, label='Load')
