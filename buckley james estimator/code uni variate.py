# Choose a single variable for univariate analysis
variable_of_interest = 'AGE'

# Fit the Cox proportional hazards model with the chosen variable
cph_univariate = CoxPHFitter(penalizer=0.1)
cph_univariate.fit(buckley_james_data[[variable_of_interest, 'Months', 'DEATH']], 'Months', 'DEATH', show_progress=True)

# Print the estimated coefficients (summary)
print(cph_univariate.summary)

# Access other properties of the fitted model (e.g., hazard ratios, p-values)
# For example, to get the hazard ratios:
print(cph_univariate.hazard_ratios_)

# Calculate AIC and BIC
n_univariate = len(buckley_james_data)
llf_univariate = cph_univariate.log_likelihood_
k_univariate = cph_univariate.params_.shape[0]
aic_univariate = -2 * llf_univariate + 2 * k_univariate
bic_univariate = -2 * llf_univariate + k_univariate * np.log(n_univariate)



# Make predictions using the univariate model
# For example, to predict the survival probability at a specific time point for a new patient:
new_patient_data_univariate = pd.DataFrame({variable_of_interest: [90], 'Months': [12], 'DEATH': [0]})
partial_hazard_univariate = cph_univariate.predict_partial_hazard(new_patient_data_univariate)
survival_prob_univariate = 1 - cph_univariate.baseline_survival_

# Plot the survival curve for the univariate model
plt.plot(cph_univariate.baseline_survival_.index, survival_prob_univariate.values)
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Survival Curve (Univariate)')
plt.show()

# Print AIC and BIC
print("AIC (univariate):", aic_univariate)
print("BIC (univariate):", bic_univariate)