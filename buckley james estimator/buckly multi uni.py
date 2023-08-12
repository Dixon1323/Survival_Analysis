import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
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

# Perform univariate analysis
univariate_results = []
for col in buckley_james_data.columns:
    if col not in ['Months', 'DEATH', 'AGE']:
        cph_univariate = CoxPHFitter(penalizer=0.1)
        cph_univariate.fit(buckley_james_data[[col, 'Months', 'DEATH', 'AGE']], 'Months', 'DEATH', show_progress=True)
        p_value = cph_univariate.summary['p'][col]
        univariate_results.append((col, p_value))
        n = len(buckley_james_data)
        llf = cph_univariate.log_likelihood_
        k = cph_univariate.params_.shape[0]
        aic = -2 * llf + 2 * k
        bic = -2 * llf + k * np.log(n)
        print(f"AIC value of {col}:", aic)
        print(f"BIC value of {col}:", bic)

# Select significant variables
significant_variables = [var for var, p_value in univariate_results if p_value < 0.05]
print("Significant variables from univariate analysis:", significant_variables)

# Fit the multivariate model
cph_multivariate = CoxPHFitter(penalizer=0.1)
cph_multivariate.fit(buckley_james_data[['Months', 'DEATH', 'AGE'] + significant_variables], 'Months', 'DEATH', show_progress=True)

# Calculate AIC and BIC
n = len(buckley_james_data)
llf = cph_multivariate.log_likelihood_
k = cph_multivariate.params_.shape[0]
aic = -2 * llf + 2 * k
bic = -2 * llf + k * np.log(n)

# Print AIC and BIC
print("AIC value of the multivariate model:", aic)
print("BIC value of the multivariate model:", bic)

# Plot the survival probability
cph_multivariate.plot()
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('James Buckley Estimator')
plt.show()
