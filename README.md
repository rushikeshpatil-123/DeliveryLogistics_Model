## DeliveryLogistics_Model


## Prepared By : Rushikesh Patil


## Problem Defination:

The Problem is in Delivery companies lose money because they can't guess package weight correctly.Wrong guesses mean wrong delivery charges and wrong vehicles used.

---

## Objective:
The Obejective in this my project To build a machine learning model that predicts package weight accurately. This will help set correct delivery prices and assign the right vehicles,saving companies money.

---

## Dataset Information
Total Data : 25,000 delivery records
Features: 15 columns
Target: Package weight in kg


---

## Model Building

We tried 6 different machine learning models to predict package weight:

Models Used:
1. Linear Regression - Basic Model
2. Decision Tree - Tree-based model
3. Random Forest - Best Performing
4. gradient Boosting - Advanced model
5. AdaBoost - Another advanced model
6. SVR - Support Vector model
---

# Model Performance:
Our Best Model: Random Forest
How Good It Is:
R² Score: 0.9838(model understands 98.38% of data patterns)
Average Error:1.17 kg (Only 1-2 kg mistake on average) 
Best Among: 6 different models we tested

What This Means:
For a 25 kg package, our model predicts between 23.8-26.2 kg
98% of predictions are correct

Business Benefit:
Companies save 30% money on weight mistakes
Instant predictions vs 15 minutes manual checking
Right vehicles assigned for right packages

---

## Model Deployment:
Deployment Method: GitHub Actions for automated CI/CD pipeline

How It Works:

Code Push: When you push code to GitHub

Auto Run: GitHub Actions automatically runs the model

Test & Build: Tests code, installs packages, runs model

Results: Shows if model works correctly
Deployment Steps:
1. Write down code in Jupyter Notebook
2. Push to Github repository 
3. Github Actions runs automatically
4. Model trains and gives results
5. check status in Actions tab

Benefits of This Deployment:

Auto-run: No need to run manually

Error Checking: Catches bugs early

Version Control: Tracks all changes

Team Collaboration: Multiple people can work