import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

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
cph = CoxPHFitter(penalizer=0.1)
cph.fit(buckley_james_data, 'Months', 'DEATH', show_progress=True)
print(cph.summary)



concordance_values = {}
for column in cph.summary.index:
    if column != 'ID':
        concordance_values[column] = concordance_index(buckley_james_data[column], cph.predict_partial_hazard(buckley_james_data))
print("Concordance values of Univariate Variables:\n")
print(concordance_values)



univariate_results = []
for col in data.columns:
    if col not in ['Months', 'ID','DEATH']:
        cph_univariate = CoxPHFitter(penalizer=0.1)
        cph_univariate.fit(data[[col, 'Months', 'DEATH']], 'Months', 'DEATH', show_progress=True)
        univariate_results.append((col, cph_univariate.summary))
        n = len(data)
        llf = cph_univariate.log_likelihood_
        k = cph_univariate.params_.shape[0]
        aic = -2 * llf + 2 * k
        bic = -2 * llf + k * np.log(n)
        univariate_aic_bic.append((col, aic, bic))
        print(f"\nAIC value of {col}:", aic)
        print(f"BIC value of {col}:", bic)

# Print the summaries of the univariate analysis
for col, summary in univariate_results:
    print(f"Univariate analysis of: {col}")
    print(summary)
    print("\n")



# Print AIC and BIC for univariate models
print("\nAIC and BIC for univariate models:")
for col, aic, bic in univariate_aic_bic:
    print(f"{col}: AIC={aic}, BIC={bic}")




significant_variables_multivariate = [(var, summary) for var, summary in multivariate_results if summary['p'][var] < 0.05]
print("\nSignificant variables from univariate analysis:")
for var, summary in significant_variables_multivariate:
    print(f"\n{var}:")
    print(summary)




#significant_variables = [(var, p_value) for var, p_value in multivariate_results if p_value < 0.05]
print("Updated data with significant variables as categorical data:")
print(data)

#data_encoded = pd.get_dummies(data, columns=[var for var, _ in significant_variables], drop_first=True)

# Update the Buckley-James data with the new categorical variables
categorical_columns = ['SEX_', 'CompositeStage_', 'LNInvolment_', 'Comorbidity_', 'FamiliyHistoryOfCancer_']
buckley_james_data = data_encoded[['Months', 'DEATH', 'AGE'] + [col for col in data_encoded.columns if any(col.startswith(cat_col) for cat_col in categorical_columns)]]





cph_multivariate = CoxPHFitter(penalizer=0.1)
cph_multivariate.fit(buckley_james_data[['Months', 'DEATH', 'AGE'] + [var for var, _ in significant_variables]], 'Months', 'DEATH', show_progress=True)
print(cph_multivariate.summary)




concordance_dict = {}
for var in cph_multivariate.params_.index:
    concordance = cph_multivariate.concordance_index_
    concordance_dict[var] = concordance
print("Concordance values of Multivariate Variables:\n")
print(concordance_dict)




n = len(buckley_james_data)
llf = cph_multivariate.log_likelihood_
k = cph_multivariate.params_.shape[0]
multivariate_aic = -2 * llf + 2 * k
multivariate_bic = -2 * llf + k * np.log(n)
print(cph_multivariate.summary)




# Print AIC and BIC for multivariate model
print("\nAIC value of the multivariate model:", multivariate_aic)
print("BIC value of the multivariate model:", multivariate_bic)

