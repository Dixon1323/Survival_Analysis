import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Load the data from the .xlsx file
data = pd.read_excel('data1.xlsx')

# Define categorical variables
categorical_cols = ['SEX', 'CompositeStage', 'LNInvolment', 'Comorbidity', 'FamiliyHistoryOfCancer']
data[categorical_cols] = data[categorical_cols].astype('category')

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Standardize the covariates
scaler = StandardScaler()
data_encoded[['DEATH', 'AGE']] = scaler.fit_transform(data_encoded[['DEATH', 'AGE']])
buckley_james_data = data_encoded[['Months', 'DEATH', 'AGE'] + [col for col in data_encoded.columns if col.startswith('SEX_') or col.startswith('CompositeStage_') or col.startswith('LNInvolment_') or col.startswith('Comorbidity_') or col.startswith('FamiliyHistoryOfCancer_')]]

# Fit the Buckley-James model with custom options
cph = CoxPHFitter(penalizer=0.1)
cph.fit(buckley_james_data, 'Months', 'DEATH', show_progress=True)
print(cph.summary)
cph.plot()
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('James Buckley Estimator')
plt.show()
n = len(buckley_james_data)
llf = cph.log_likelihood_
k = cph.params_.shape[0]
aic = -2 * llf + 2 * k
bic = -2 * llf + k * np.log(n)
# Print AIC and BIC
print("AIC value of the above data:", aic)
print("BIC value of the above data:", bic)