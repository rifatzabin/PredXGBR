
from packages import * #import all modules and libraries


# Load and preprocess data
pjme = pd.read_csv('../../../PJM_Load/PJM_Load_hourly.csv', index_col=[0], parse_dates=[0])



# Define the date to split the dataset into training and testing
split_date = '07-Aug-2000'
df_train = pjme.loc[pjme.index <= split_date].copy()
df_test = pjme.loc[pjme.index > split_date].copy()



def create_features(df, label=None):
    """
    %%Creates time series features from datetime index
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

    # Lag and rolling window features based on 'Load'
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

    # Extract features for model training
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear' , 'pjme_6_hrs_lag' , 'pjme_24_hrs_lag' , 'pjme_6_hrs_mean',
           "pjme_12_hrs_mean" ,"pjme_24_hrs_mean" ,"pjme_6_hrs_std" ,"pjme_12_hrs_std" ,"pjme_24_hrs_std",
           "pjme_6_hrs_max","pjme_12_hrs_max" ,"pjme_24_hrs_max" ,"pjme_6_hrs_min","pjme_12_hrs_min" ,"pjme_24_hrs_min"]]
    if label:
        y = df[label] # If a label is provided, extract it for supervised learning
        return X, y
    return X

# Create feature sets and labels for training and testing datasets
X_train, y_train = create_features(df_train, label='Load')
X_test, y_test = create_features(df_test, label='Load')
