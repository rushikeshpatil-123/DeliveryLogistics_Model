
# Data Manipulation Libaries
import numpy as np
import pandas as pd

# Data Visiualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Import Basic Warning Libraries 
import warnings
warnings.filterwarnings(action='ignore')

# Import Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
from collections import OrderedDict
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV,cross_val_score,KFold

# Import Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

filepath = "https://raw.githubusercontent.com/rushikeshpatil-123/DeliveryLogistics_Model/refs/heads/main/data/raw/Delivery_Logistics.csv" \

# Function Defination

# Step 1: data ingestion

def data_ingestion():
    return pd.read_csv(filepath)

# step 2: Data Exploration 

def data_exploration(df):

    numerical_col = df.select_dtypes(exclude = 'object').columns
    categorical_col = df.select_dtypes(include = 'object').columns

    num_stats_list = []
    cat_stats_list = []

    # Numerical Features-----------------------
    for col in numerical_col:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR

        outlier_count = ((df[col] < LW) | (df[col] > UW)).sum()

        stats = OrderedDict({
        "Feature": col,
        "Mean" : df[col].mean(),
        "Medain":df[col].median(),
        "Maximum":df[col].max(),
        "Minimum":df[col].min(),
        "Q1":Q1,
        "Q3":Q3,
        "IQR": IQR,
        "Lower_Limit":LW,
        "Upper_Limit":UW,
        "Outlier_Count":outlier_count,
        "Skewness":df[col].skew(),
        "Kurtosis":df[col].kurt(),
    })

    num_stats_list.append(stats)

    numerical_stats_report = pd.DataFrame(num_stats_list)
   

         # -------------------------
    # Categorical Features
    # -------------------------
    for col in categorical_col:
        cat_stats = OrderedDict({
            "Feature": col,
            "Unique Values": df[col].nunique(),
            "Mode": df[col].mode()[0],
            "Missing Values": df[col].isnull().sum(),
            "Value Counts": df[col].value_counts().to_dict()
        })

        cat_stats_list.append(cat_stats)

    categorical_stats_report = pd.DataFrame(cat_stats_list)

    # -------------------------
    # Dataset Info
    # -------------------------
    dataset_info = pd.DataFrame({
        "Feature": df.columns,
        "Dtype": df.dtypes.values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    return numerical_stats_report, categorical_stats_report, dataset_info

def data_preprocessing(df):

    X = df.drop(columns = ['package_weight_kg'],axis = 1)
    y = df['package_weight_kg']

    # Split the Dataset into train and test

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size= 0.3,
                                                        random_state= 0)
    
    # Use Encoding technique to convert all categorical columns into numerical columns\

    categorical_col = X.select_dtypes(include = 'object').columns


    from sklearn.preprocessing import LabelEncoder,MinMaxScaler
    for i in categorical_col:
        le = LabelEncoder()
        X_train[i] = le.fit_transform(X_train[i])
        X_test[i] = le.transform(X_test[i])

    # Using Normaliazation Technique
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)  # Seen Data
    X_test = sc.transform(X_test)        # Unseen Data
    return X_train , X_test , y_train , y_test

# Step 4: Model building
def model_building(X_train, X_test, y_train, y_test):

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegression": DecisionTreeRegressor(),
        "RandomForestRegression": RandomForestRegressor(),
        "GradientBoostRegression": GradientBoostingRegressor(),
        "AdaBoostRegression": AdaBoostRegressor(),
        "SVR": SVR(),
    }

    Regression_models = []

    for model_name, model in models.items():

        print(f"Training model: {model_name}")  # 🔍 CMD debug

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        r2_score_model = r2_score(y_test, y_pred)
        mse_model = mean_squared_error(y_test, y_pred)
        mae_model = mean_absolute_error(y_test, y_pred)

        Regression_models.append({
            "model name": model_name,
            "R2 Score": r2_score_model,
            "Mse": mse_model,
            "Mae": mae_model
        })

    
    Regression_models_report = pd.DataFrame(Regression_models)
    return Regression_models_report



# Function Calling 

# step1 : Data Ingestion
df = data_ingestion()

# step2: Data Exploration
numerical_stats_report, categorical_stats_report, dataset_info = data_exploration(df)

# step3: Data Preprocessing
X_train, X_test, y_train, y_test = data_preprocessing(df)

# step4: Model Building
model_report = model_building(X_train, X_test, y_train, y_test)

# Use RandomForest with GridSearch CV 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Build Model with RandomForest 

rf = RandomForestRegressor(random_state=42)

# Hyperparameter grid

param_grid = {
    'n_estimators': [100],
    'max_depth':[None]

}

# GridSearch CV
grid = GridSearchCV(
    estimator = rf,
    param_grid = param_grid,
    cv = 5,
    n_jobs= -1,
    verbose= 1
)
# Fit On training Data 
grid.fit(X_train,y_train)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

print(df)
print(numerical_stats_report)
print(categorical_stats_report)
print(dataset_info)
print(model_report)
