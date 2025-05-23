

from packages import * # Import necessary Python libraries
from datagen import *  # Import data generation utilities



# Initialize the XGBoost regressor with a specified number of estimators
reg = xgb.XGBRegressor(n_estimators=1000) # Set up the regressor with 1000 trees
# Fit the model on the training dataset with early stopping
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)], # Provide both training and testing sets for evaluation
        early_stopping_rounds=50, # Stop training if evaluation metric does not improve for 50 rounds
       verbose=True) # Change verbose to True if you want to see it train

# Save the trained model using pickle for later use or deployment
pickle.dump(reg, open('xgboost_model2_pjm_load_hourly.model', 'wb')) # Using a context manager ensures the file is properly closed after writing





