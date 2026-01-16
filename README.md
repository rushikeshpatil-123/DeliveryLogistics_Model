## DeliveryLogistics_Model


## Prepared By : Rushikesh Patil



# Problem Defination

In real-world logistics systems, package weight plays a critical role in determining delivery cost, vehicle allocation, delivery mode, and overall operational efficiency. However, package weight information may sometimes be missing, delayed, or inaccurate.

To address this challenge, this project formulates the problem as a supervised machine learning regression task, where the goal is to predict the package weight (in kilograms) using delivery and logistics-related features.

---

# Dataset Information
Dataset Type: Structured CSV file

Total Records: ~25000 delivery entries

Target Variable: package_weight_kg

Data Quality: No missing values

Outliers : No Outliers Present


---

# Key Features

Numerical Features:

distance_km
package_weight_kg
delivery_rating
delivery_cost

Categorical Features:

delivery_partner
package_type
vehicle_type
delivery_mode
region
weather_condition

The dataset is clean and well-structured, making it suitable for regression modeling.

---

# Key Insights(EDA)

Numerical Analysis

Calculated mean, median, quartiles (Q1, Q3), IQR, skewness, and kurtosis
Outlier detection performed using the IQR method
No significant outliers observed

Categorical Analysis

Analyzed unique values, modes, and frequency distributions
Understood delivery patterns across regions, vehicle types, and weather conditions

Weight Category Analysis

Package weights were grouped into three categories:
Light (0–10 kg)
Medium (10–25 kg)
Heavy (25+ kg)

Crosstab analysis revealed that:

Heavy packages are mostly delivered using vans and trucks
Delivery mode and vehicle type vary significantly with package weight

Data Preprocessing

Train-Test Split: 70% training and 30% testing
Encoding: Label Encoding applied to all categorical variables
Feature Scaling: MinMaxScaler used to normalize features between 0 and 1
These steps ensured that the data was suitable for machine learning algorithms and improved model stability.

---

# Model Building

Multiple regression models were trained and compared:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

AdaBoost Regressor

Support Vector Regressor (SVR)

This comparative approach helped identify the most suitable algorithm for the dataset.


---

# Model Performance:

Model performance was evaluated using:

R² Score

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Results Summary

Random Forest Regressor achieved the best performance with an R² score of ~98%

Decision Tree Regressor achived the performance with an r2 score of ~95%

Linear Regression achieved an R² score of ~66%

The superior performance of Random Forest highlights its ability to capture non-linear relationships in logistics data

---

# Hyperparameter Tuning

Applied GridSearchCV on Random Forest Regressor

Tuned parameters such as number of trees (n_estimators) and tree depth (max_depth)

Achieved a cross-validated R² score of approximately 0.98, confirming model stability and reduced overfitting

---

# MLOps Integration (GitHub Actions)

Implemented GitHub Actions for Continuous Integration (CI)

Automated workflow runs on every code push

CI Pipeline Steps:

Set up Python environment

Install required dependencies

Execute the main ML pipeline script

Validate successful execution without errors

This automation ensures reproducibility, reliability, and early error detection.

# Conclusion

This project demonstrates a complete end-to-end machine learning workflow, including:

Data ingestion and exploration

Feature engineering and preprocessing

Model training and evaluation

Hyperparameter tuning

CI automation using GitHub Actions

The final model can assist logistics companies in delivery planning, cost estimation, and resource optimization, while also showcasing practical ML and basic MLOps skills.

---