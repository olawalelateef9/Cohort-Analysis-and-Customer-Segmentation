import mlflow
import mlflow.sklearn
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the refined dataset
df = pd.read_csv(r"C:\Users\User\Downloads\refined_jewelry_data.csv")

# Separate features and target
X = df.drop(columns=['Price_USD'])  
y = df['Price_USD']

# Encode categorical features
categorical_cols = ['Category', 'Target_Gender', 'Main_Color', 'Main_Metal', 'Main_Gem']
for col in categorical_cols:
    X[col] = X[col].astype(str)  # Convert to string in case of NaN values
    X[col] = LabelEncoder().fit_transform(X[col])



# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)


imputer = SimpleImputer(strategy="most_frequent")  
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)



# Initialize MLflow experiment
experiment_name = "Jewelry Price Optimization"
mlflow.set_experiment(experiment_name)

# Define and train models
cat_model = CatBoostRegressor()
cat_model.fit(X_train, y_train)  

lin_pipe = LinearRegression()
lin_pipe.fit(X_train, y_train)

ada_pipe = AdaBoostRegressor()
ada_pipe.fit(X_train, y_train)

extra_pipe = ExtraTreesRegressor()
extra_pipe.fit(X_train, y_train)

# Dictionary of models and their metrics 
models = {
    "CatBoost": {
        "r2_train": 0.33559,
        "r2_test": -0.14593,
        "rmse_train": 365.16,
        "rmse_test": 387.58,
        "model": cat_model  
    },
    "Linear Regression": {
        "r2_train": 0.15764,
        "r2_test": 0.16061,
        "rmse_train": 411.17,
        "rmse_test": 331.72,
        "model": lin_pipe  
    },
    "AdaBoost": {
        "r2_train": 0.14712,
        "r2_test": 0.04438,
        "rmse_train": 413.73,
        "rmse_test": 353.94,
        "model": ada_pipe  
    },
    "ExtraTrees": {
        "r2_train": 0.32080,
        "r2_test": 0.15715,
        "rmse_train": 369.20,
        "rmse_test": 332.40,
        "model": extra_pipe  
    }
}

# Iterate over models and log experiments
for model_name, metrics in models.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("r2_train", metrics["r2_train"])
        mlflow.log_metric("r2_test", metrics["r2_test"])
        mlflow.log_metric("rmse_train", metrics["rmse_train"])
        mlflow.log_metric("rmse_test", metrics["rmse_test"])
        
        # Log model artifact
        mlflow.sklearn.log_model(metrics["model"], model_name)

        print(f"Logged {model_name} in MLflow")

print("Experiment tracking completed!")

