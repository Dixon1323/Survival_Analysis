import pandas as pd
from lifelines import CoxPHFitter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_excel('data1.xlsx') # Load the data from the .xlsx file

# Preprocess the data Drop any rows with missing values in the columns of interest
data = data.dropna(subset=['Months', 'DEATH', 'AGE', 'SEX', 'CompositeStage', 'LNInvolment', 'Comorbidity', 'FamiliyHistoryOfCancer'])
# Handle missing values in other columns
imputer = SimpleImputer(strategy='median')
data[['DEATH', 'AGE', 'CompositeStage', 'LNInvolment', 'Comorbidity']] = imputer.fit_transform(data[['DEATH', 'AGE', 'CompositeStage', 'LNInvolment', 'Comorbidity']])
# Standardize the covariates
scaler = StandardScaler()
data[['DEATH', 'AGE', 'CompositeStage', 'LNInvolment', 'Comorbidity']] = scaler.fit_transform(data[['DEATH', 'AGE', 'CompositeStage', 'LNInvolment', 'Comorbidity']])
# Create a new DataFrame with the required columns for the Buckley-James estimator
buckley_james_data = data[['Months', 'DEATH', 'AGE', 'SEX', 'CompositeStage', 'LNInvolment', 'Comorbidity', 'FamiliyHistoryOfCancer']]
#print(buckley_james_data.isnull().sum()) #find no of Nan
# Fit the Buckley-James model with custom options
#buckley_james_data = buckley_james_data.drop(['tstart'], axis=1)
#buckley_james_data = buckley_james_data.drop('tstart', axis=1)
cph = CoxPHFitter(penalizer=0.1)  # Set the penalizer parameter to control overfitting
cph.fit(buckley_james_data, 'Months', 'DEATH', show_progress=True)  # Set the step_size parameter to control the convergence speed
# Print the estimated coefficients
print(cph.summary)
# Access other properties of the fitted model (e.g., hazard ratios, p-values)
# For example, to get the hazard ratios:
print(cph.hazard_ratios_)
# Make predictions using the fitted model
# For example, to predict the survival probability at a specific time point for a new patient:
new_patient_data = pd.DataFrame({'AGE': [90], 'SEX': [1], 'CompositeStage': [2], 'LNInvolment': [1], 'Comorbidity': [0], 'FamiliyHistoryOfCancer': [0]})
partial_hazard = cph.predict_partial_hazard(new_patient_data)
survival_prob = 1 - cph.baseline_survival_
plt.plot(cph.baseline_survival_.index, survival_prob.values)
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Survival Curve')
plt.show()
# Perform other analyses or visualizations as needed