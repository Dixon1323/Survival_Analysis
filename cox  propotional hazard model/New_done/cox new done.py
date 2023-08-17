import pandas as pd
from lifelines import CoxPHFitter
import numpy as np

# Read the Excel file into a pandas DataFrame
data = pd.read_excel('DATA1.xlsx')

# Create a CoxPHFitter object
cph = CoxPHFitter()

# Fit the Cox proportional hazards model
cph.fit(data, duration_col='Months', event_col='DEATH')

# Print the summary of the model
print(cph.summary)




univariate_results = []
univariate_aic_bic = []
for col in data.columns:
    if col not in ['Months', 'ID']:
        cph_univariate = CoxPHFitter(penalizer=0.1)
        cph_univariate.fit(data[['Months', 'ID', col]], duration_col='Months', event_col='ID', show_progress=True)
        univariate_results.append((col, cph_univariate.print_summary()))




for col in data.columns:
    if col not in ['Months', 'ID']:
        n = len(data)
        llf = cph_univariate.log_likelihood_
        k = cph_univariate.params_.shape[0]
        aic = -2 * llf + 2 * k
        bic = -2 * llf + k * np.log(n)
        univariate_aic_bic.append((col, aic, bic))
        print(f"\nColumn: {col}")
        print(f"AIC value: {aic}")
        print(f"BIC value: {bic}")




# Calculate p-values for each variable
p_values = []
for col in data.columns:
    if col not in ['Months', 'DEATH']:
        cph_univariate = CoxPHFitter(penalizer=0.1)
        cph_univariate.fit(data[['Months', 'DEATH', col]], duration_col='Months', event_col='DEATH', show_progress=True)
        p_values.append((col, cph_univariate.summary['p'][col]))

# Sort the p-values list in ascending order
p_values.sort(key=lambda x: x[1])

# Get the significant variable with the lowest p-value
significant_variable_pvalue = p_values[0][0]
significant_variable_pvalue_value = data[significant_variable_pvalue].iloc[0]
print(f"\nSignificant variable based on p-value: {significant_variable_pvalue}")
#print(f"Value of the significant variable: {significant_variable_pvalue_value}")



cph_univariate = CoxPHFitter(penalizer=0.1)
cph_univariate.fit(data[['Months', 'DEATH', significant_variable_pvalue]], duration_col='Months', event_col='DEATH', show_progress=True)
univariate_results = cph_univariate.print_summary()

# Print the univariate analysis result
print(univariate_results)




n = len(data)
llf = cph_univariate.log_likelihood_
k = cph_univariate.params_.shape[0]
aic = -2 * llf + 2 * k
bic = -2 * llf + k * np.log(n)

print(f"AIC value: {aic}")
print(f"BIC value: {bic}")
