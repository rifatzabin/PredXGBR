

from packages import *
from datagen import * 



reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=True) # Change verbose to True if you want to see it train

pickle.dump(reg, open('xgboost_model2_AEP_hourly.model', 'wb'))





