from packages import * # Imports all modules and libraries
from datagen import * # Imports data generation utilities


# Loading a pre-trained XGBoost model
reg = pickle.load(open('xgboost_model2_pjm_load_hourly.model', 'rb'))


# Visualize feature importance from the XGBoost model
ax3= plot_importance(reg, height=0.75, color='green', max_num_features = 10)
ax3.set_facecolor('white')
plt.xlabel("F Score", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.title("Feature Importance", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()




# Predict and append the results to the test dataset
df_test['MW_Prediction'] = reg.predict(X_test)
pjme_all = pd.concat([df_test, df_train], sort=False)


# Plot prediction results along with actual data
ax4 = pjme_all[['Load','MW_Prediction']].plot(figsize=(15, 5), color=['green', 'orange'], style='.')
ax4.set_facecolor('white')
plt.xlabel("Time Series", fontsize=19)
plt.ylabel("Electrical Load (W)", fontsize=19)
plt.title("PJM Load", fontsize=21)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.show()


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','Load']].plot(ax=ax, style=['-','.'], color=['orange', 'green'])
ax.set_xbound(lower='01-01-2001', upper='02-01-2001')
ax.set_ylim(10000, 50000)
plot = plt.suptitle('January 2001 Forecast vs Actuals', fontsize=17)
ax.set_facecolor('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()




# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','Load']].plot(ax=ax, style=['-','.'], color=['orange', 'green'])
ax.set_xbound(lower='01-01-2001', upper='01-08-2001')
ax.set_ylim(10000, 50000)
plot = plt.suptitle('First Week of January Forecast vs Actuals', fontsize=17)
ax.set_facecolor('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()




f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','Load']].plot(ax=ax, style=['-','.'], color=['orange', 'green'])
ax.set_ylim(10000, 50000)
ax.set_xbound(lower='07-01-2001', upper='07-08-2001')
plot = plt.suptitle('First Week of July Forecast vs Actuals')
plot = plt.suptitle('First Week of July Forecast vs Actuals', fontsize=17)
ax.set_facecolor('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()





# Evaluate model performance using R^2, MSE, MAE, and MAPE
output = reg.predict(X_test)
xgb_score = r2_score(y_test, output)
print("R^2 Score of XGBoost model = ",xgb_score)



# Calculate and print Mean Squared Error
mean_squared_error = mean_squared_error(y_true=df_test['Load'],
                   y_pred=df_test['MW_Prediction'])
print("Mean_squared_error =", mean_squared_error)

# Calculate and print Mean Absolute Error
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