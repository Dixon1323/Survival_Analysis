import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# Load the data from the Excel file
data = pd.read_excel('data1.xlsx')

# Create a new instance of the CoxPHFitter class
cph = CoxPHFitter()

# Fit the Cox Proportional Hazard model to the data
cph.fit(data, duration_col='Months', event_col='DEATH')

covariates_to_plot = ['AGE', 'SEX', 'CompositeStage', 'LNInvolment', 'Comorbidity', 'FamiliyHistoryOfCancer']
significant_covariates = []

for covariate in covariates_to_plot:
    cph.plot_partial_effects_on_outcome(covariates=covariate, values=[0, 1], cmap='coolwarm')
    plt.xlabel(covariate)
    plt.ylabel('Survival Probability')
    plt.title('Survival Curves by ' + covariate)
    plt.legend(['0', '1'])
    plt.show()
    
    # Prompt user to input whether the covariate is significant or not
    is_significant = input("Is " + covariate + " significant? (y/n): ")
    
    if is_significant.lower() == 'y':
        significant_covariates.append(covariate)

# Perform multivariate analysis
multivariate_data = data[significant_covariates + ['Months', 'DEATH']]
cph_multivariate = CoxPHFitter()
cph_multivariate.fit(multivariate_data, duration_col='Months', event_col='DEATH')

n = len(data)
llf_univariate = cph.log_likelihood_
k_univariate = cph.params_.shape[0]
aic_univariate = -2 * llf_univariate + 2 * k_univariate
bic_univariate = -2 * llf_univariate + k_univariate * np.log(n)

llf_multivariate = cph_multivariate.log_likelihood_
k_multivariate = cph_multivariate.params_.shape[0]
aic_multivariate = -2 * llf_multivariate + 2 * k_multivariate
bic_multivariate = -2 * llf_multivariate + k_multivariate * np.log(n)

print("Univariate Analysis:")
print("Significant Covariates:", significant_covariates)
print("AIC of the univariate model:", aic_univariate)
print("BIC of the univariate model:", bic_univariate)

print("\nMultivariate Analysis:")
print(cph_multivariate.summary)
print("AIC of the multivariate model:", aic_multivariate)
print("BIC of the multivariate model:", bic_multivariate)
