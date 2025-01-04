### The below syntax chunk observes importing relevant libraries & loading my dataset into Python, Additionally naming our report
import streamlit as st # To build report
import matplotlib.pyplot as plt # For plotting visualisations
import pandas as pd # For data manipulation, easy to use data structures, reading the loaded CSV file and statistical summaries
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, precision_recall_curve, auc, matthews_corrcoef #
import statsmodels.api as sm # For additional statistical support
import numpy as np # Loaded to handle numerical operations within the syntax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle # To save and load files
import seaborn as sns # for plotting visualisations
from statsmodels.stats.outliers_influence import variance_inflation_factor # To create VIF for independent variables
st.title ('Claims Cause Model Performance Dashboard')
Claims = pd.read_csv("C:\\Users\\Ageas\\Documents\\Data Science Degree\\Software Engineering\\Insurance_Claims_Data.csv")
print(Claims) 
# The below Syntax chunk observes checking the dataset for null values and transforming selected columns to change from string to boolean and change them to lowercase,
# this is to ensure if any other columns have identical string values that this is not replaced. Although not the most efficiant way, this is to ensure error prevention
#  and clarity of code for users. 
null_values = Claims.isnull().sum()
print(null_values)
columns_to_replace = [
    'is_parking_camera', 'is_esc', 'is_adjustable_steering', 'is_tpms', 
    'is_parking_sensors', 'is_front_fog_lights', 'is_rear_window_wiper', 
    'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist', 
    'is_power_door_locks', 'is_central_locking', 'is_power_steering', 
    'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror', 
    'is_ecw', 'is_speed_alert'
]
for column in columns_to_replace:
    Claims[column] = Claims[column].str.lower().map({'yes': 1, 'no': 0})
print(Claims)
# The below syntax chunk observes preparing the dataset for the model by selecting the specified columns to use as variables, I am doing this by creating a new data frame called 
# 'Claims_Model'
Claims_Model = Claims[['subscription_length', 'vehicle_age', 'customer_age', 'is_central_locking', 'is_parking_sensors', 'is_parking_camera', 'claim_status']]
print(Claims_Model.head())
print(Claims_Model)
# The below syntax chunk observes creating a correlation matrix to check to multicollinearity and visualise it with a correlation heatmap. It shows the plotting of the 
# the creation of a dataframe with 2 columns: 'Feature' which outlines all variables and 'VIF' which produces the corresponding values. Then it shows the the use of plotly
# to create the heatmap, customising the size, palette and layout. Finally, it shows the calculation of the Variance Inflation Factor (VIF) for each independent variable,
#  'X' represents the intercept

correlation_matrix = Claims_Model.corr()
print("Correlation Matrix:\n", correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

X = Claims_Model[['subscription_length', 'vehicle_age', 'customer_age', 'is_central_locking', 'is_parking_sensors', 'is_parking_camera']]
X = sm.add_constant(X)  

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("Variance Inflation Factor:\n", vif_data)
# The below syntax chunk observes the creation of our predictive model by defining the independent variables (X) and the dependent variable (y) for our model. It then shows
# the fitting of the model, setting perameters for predicting probabilities.
X = Claims_Model[['subscription_length', 'vehicle_age', 'customer_age', 'is_central_locking', 'is_parking_sensors', 'is_parking_camera']]
y = Claims_Model['claim_status']
X = sm.add_constant(X)

logit_claims_model = sm.Logit(y, X).fit()
print(logit_claims_model.summary())

claims_predict = logit_claims_model.predict(X)

fpr_logit, tpr_logit, thresholds = roc_curve(y, claims_predict)
auc_logit = roc_auc_score(y, claims_predict)

plt.figure(figsize=(10, 8))
plt.plot(fpr_logit, tpr_logit, label=f'Logistic Regression (AUC = {auc_logit:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='r')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
# The below syntax chunk shows a retesting of the model using random forest to compare models. This evidences the fitting of the model and the test train split as well as
# repeating the steps to predict the probabilities and calculation of the receiver operating characteristic curve (ROC) and area under the curve (AUC)
X = Claims_Model[['subscription_length', 'vehicle_age', 'customer_age', 'is_central_locking', 'is_parking_sensors', 'is_parking_camera']]
y = Claims_Model['claim_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_predict = rf_model.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, thresholds = roc_curve(y_test, rf_predict)
auc_rf = roc_auc_score(y_test, rf_predict)

print("Feature Importances:")
for feature, importance in zip(X.columns, rf_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

y_pred = rf_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='r')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
# Finally, we visualise the calculation of the precision recall curve for both logistic regression and random forest models. Additionally, I created a Matthews correlation 
# coefficient to validate the model knowing the classes are imbalanced. Finally, we used Pickle to save our model prior to deployment.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logit_claims_model = sm.Logit(y_train, sm.add_constant(X_train)).fit()
logit_predict = logit_claims_model.predict(sm.add_constant(X_test))
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict_proba(X_test)[:, 1]
precision_logit, recall_logit, _ = precision_recall_curve(y_test, logit_predict)
auc_logit = auc(recall_logit, precision_logit)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_predict)
auc_rf = auc(recall_rf, precision_rf)
claims_pred_binary = [1 if prob > 0.5 else 0 for prob in claims_predict]
mcc = matthews_corrcoef(y, claims_pred_binary)
print(f"Matthews Correlation Coefficient: {mcc}")


plt.figure(figsize=(10, 8))
plt.plot(recall_logit, precision_logit, label=f'Logistic Regression (AUC = {auc_logit:.2f})')
plt.plot(recall_rf, precision_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

with open('logit_claims_model.pkl', 'wb') as file:
    pickle.dump(logit_claims_model, file)
# Here is where we build our plots for the dashboard using all of the components built above.
st.header('ROC Curve')
fig, ax = plt.subplots()
ax.plot(fpr_logit, tpr_logit, label=f'Logistic Regression (AUC = {auc_logit:.2f})')
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
ax.plot([0, 1], [0, 1], linestyle='--', color='r')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)
st.header('Precision-Recall Curve')
fig, ax = plt.subplots()
ax.plot(recall_logit, precision_logit, label=f'Logistic Regression (AUC = {auc_logit:.2f})')
ax.plot(recall_rf, precision_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
st.pyplot(fig)
st.header('Feature Importances (Random Forest)')
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
st.bar_chart(feature_importances)

st.header('Classification Report')
st.text('Logistic Regression:')
st.text(sm.Logit(y_train, sm.add_constant(X_train)).fit().summary())
st.text('Random Forest:')
y_pred_rf = rf_model.predict(X_test)
st.text(classification_report(y_test, y_pred_rf))
#1. To connect to the path to deploy Streamlit run: 
# streamlit run "c:/Users/Ageas/Documents/Data Science Degree/Software Engineering/modelreport.py" , in your terminal



