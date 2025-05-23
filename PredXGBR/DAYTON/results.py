from packages import *
from datagen import *


reg = pickle.load(open('xgboost_model2_DAYTON_hourly.model', 'rb'))


df_test['MW_Prediction'] = reg.predict(X_test)
pjme_all = pd.concat([df_test, df_train], sort=False)




output = reg.predict(X_test)
xgb_score = r2_score(y_test, output)
print("R^2 Score of XGBoost model = ",xgb_score)




mean_squared_error = mean_squared_error(y_true=df_test['Load'],
                   y_pred=df_test['MW_Prediction'])
print("Mean_squared_error =", mean_squared_error)

mean_absolute_error = mean_absolute_error(y_true=df_test['Load'],
                   y_pred=df_test['MW_Prediction'])
print("Mean_absolute_error =", mean_absolute_error)

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error = mean_absolute_percentage_error(y_true=df_test['Load'],
                   y_pred=df_test['MW_Prediction'])

print("Mean_Absolute_Percentage_Error = " + str(mean_absolute_percentage_error) + " %")