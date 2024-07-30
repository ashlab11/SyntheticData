from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_error, make_scorer
import pickle
from scipy.stats import randint, uniform, loguniform
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import loguniform
from sklearn.linear_model import Lasso, Ridge
"""
**We want the following datasets**
1. Real Training/Testing data, no missingness
2. Synthetic Training data, no missingness (test ML model on real test data)

*No imputation, XGBOOST*

1. Real Training/Testing data, MAR
2. Synthetic Training data, MAR (test ML model on real test data)
3. Real Training/Testing data, MNAR
4. Synthetic Training data, MNAR (test ML model on real test data)

*Imputation Beforehand*

1. Real Training data, MAR/MNAR then imputed
2. Real Testing data, no missing data
3. Synthetic training data, (test ML model on real test data)

*Imputation After*

1. Real training data, MAR/MNAR
2. Real testing data, no missingness
3. Synthetic training data, imputed (test ML model on real test data)

**ORGANIZING**

1. Split data into training and testing, with no missingness
2. Copy training twice and add MAR/MNAR to them
3. Copy testing twice and add MAR/MNAR to them
4. Impute training data, and synthesize it
5. Synthesize training data with missingness"""

# function for the ML pipeline
def ML_pipe_kfold(train_data : pd.DataFrame, ML_algo : any, preprocessor : ColumnTransformer, param_dist : dict, impute : bool = False):
    '''
    This function finds the best ML model using GridSearchCV with KFold cross-validation.
    Returns a model to be pickled 
    '''
    
    if impute:
        reg = make_pipeline(preprocessor, IterativeImputer(), ML_algo)
    else:
        reg = make_pipeline(preprocessor, ML_algo)
    
    X = train_data.drop(columns = ['price'])
    y = train_data['price']
    
    param_dist = {f"{ML_algo.__class__.__name__.lower()}__{k}": v for k, v in param_dist.items()} # Fixing parameter grid so it works with sklearn pipe

    folds = KFold(n_splits = 4, shuffle=True, random_state=42)
    grid = RandomizedSearchCV(reg, param_distributions=param_dist, cv = folds, scoring = 'neg_root_mean_squared_error', n_iter = 60, n_jobs = -1, verbose = 1)
    grid.fit(X, y)
    print(f"Best test score: {grid.best_score_}")
    return grid.best_estimator_


#---- PREPROCESSOR ---
one_hot_fts = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
std_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
preprocessor = ColumnTransformer(
    [('one_hot', OneHotEncoder(), one_hot_fts),
    ("std", StandardScaler(), std_features)]
)

def main(train_data : pd.DataFrame, test_data : pd.DataFrame):
    """This function runs the ML pipeline for the given dataset. We'll be running this a 
    bunch of times with different missing/no missing values datasets.
    
    Eval_type refers to the different types of evaluations we are doing. It can take the following values:
    - real: We are evaluating the model on real, no missing data
    - imputation before: We are evaluating the model on data that has been imputed before
    - imputation after: We are evaluating the model on data that will be imputed now
    - no imputation: We are evaluating the model only with XGBoost"""
    
    #---- XGBOOST ----
    param_dist = {
        'n_estimators': randint(100, 300),  # Discrete uniform distribution
        'max_depth': randint(2, 6),  # Discrete uniform distribution, max exclusive
        'learning_rate': loguniform(0.001, 0.1),  # Continuous uniform distribution
        'subsample': uniform(0.5, 0.1),  # Continuous uniform distribution from 0.5 to 1
        'colsample_bytree': uniform(0.5, 0.1)  # Continuous uniform distribution from 0.5 to 1
    }

    xgb_reg = xgb.XGBRegressor()
    #XGBoost is the only model that can handle missing values, so we will test it with and without imputing missing values
    xgb_grid_no_impute = ML_pipe_kfold(train_data, xgb_reg, preprocessor, param_dist, impute = False)
    xgb_grid_impute = ML_pipe_kfold(train_data, xgb_reg, preprocessor, param_dist, impute = True)

    #---- RANDOM FOREST ----

    # Parameter distribution for Random Forest
    rf_param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5)
    }
    rf_reg = RandomForestRegressor()
    rf_grid = ML_pipe_kfold(train_data, rf_reg, preprocessor, rf_param_dist, impute=True)

    #---- LASSO/RIDGE REGRESSION ----
    

    # Parameter distribution for Lasso
    lasso_param_dist = {
        'alpha': loguniform(1e-4, 1e3)
    }

    lasso_reg = Lasso(max_iter=10000)
    lasso_grid = ML_pipe_kfold(train_data, lasso_reg, preprocessor, lasso_param_dist, impute=True)

    # Parameter distribution for Ridge
    ridge_param_dist = {
        'alpha': loguniform(1e-4, 1e3)
    }

    ridge_reg = Ridge(max_iter=10000)
    ridge_grid = ML_pipe_kfold(train_data, ridge_reg, preprocessor, ridge_param_dist, impute=True)

    #---- KNN -----
    from sklearn.neighbors import KNeighborsRegressor

    # Parameter distribution for KNN
    knn_param_dist = {
        'n_neighbors': randint(3, 20),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    knn_reg = KNeighborsRegressor()
    knn_grid = ML_pipe_kfold(train_data, knn_reg, preprocessor, knn_param_dist, impute=True)


    #---- SVR -----

    # Parameter distribution for SVR
    svr_param_dist = {
        'C': loguniform(1e-4, 1e3),
        'gamma': loguniform(1e-4, 1e-1),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    svr_reg = SVR()
    svr_grid = ML_pipe_kfold(train_data, svr_reg, preprocessor, svr_param_dist, impute=True)
    
    results = {}

    for model in [xgb_grid_no_impute, xgb_grid_impute, rf_grid, lasso_grid, ridge_grid, knn_grid, svr_grid]:
        predictions = model.predict(test_data.drop(columns = ['price']))
        model_name = list(model.named_steps)[-1]
        print(model_name)
        if model == xgb_grid_no_impute:
            model_name += "_no_impute"
        results[model_name] = root_mean_squared_error(test_data['price'], predictions)
    
    return list(results.values())

def create_results_df():
    columns = ['run', 'data_type', 'model', 'score']
    return pd.DataFrame(columns=columns)


n_runs = 10

original_results = create_results_df()
synthpop_results = create_results_df()
synthesizer_results = create_results_df()


model_names = ['xgbregressor_no_impute', 'xgbregressor', 'randomforestregressor', 'lasso', 'ridge', 'kneighborsregressor', 'svr']
data_types = ['original', 'mar', 'mnar']

for run in range(n_runs):
    for data_type in data_types:
        # Load datasets
        original_train_data = pd.read_csv(f"Datasets/housing_{data_type}_train_{run}.csv")
        original_synthpop_train_data = pd.read_csv(f"Datasets/synthpop_housing_{data_type}_train_{run}.csv")
        original_synthesizer_train_data = pd.read_csv(f"Datasets/synthesizer_housing_{data_type}_train_{run}.csv")
        test_data = pd.read_csv(f"Datasets/housing_{data_type}_test_{run}.csv")
        
        # Run main function and store results
        original_scores = main(original_train_data, test_data)
        synthpop_scores = main(original_synthpop_train_data, test_data)
        synthesizer_scores = main(original_synthesizer_train_data, test_data)
        
        # Append results to respective DataFrames
        for model, score in zip(model_names, original_scores):
            original_results = pd.concat([original_results, pd.DataFrame({
                'run': run,
                'data_type': data_type,
                'model': model,
                'score': score
            }, index = [0])], ignore_index=True)
        
        for model, score in zip(model_names, synthpop_scores):
            synthpop_results = pd.concat([synthpop_results, pd.DataFrame({
                'run': run,
                'data_type': data_type,
                'model': model,
                'score': score
            }, index = [0])], ignore_index=True)
        
        for model, score in zip(model_names, synthesizer_scores):
            synthesizer_results = pd.concat([synthesizer_results, pd.DataFrame({
                'run': run,
                'data_type': data_type,
                'model': model,
                'score': score
            }, index = [0])], ignore_index=True)

original_results.to_csv("original_results_detailed.csv", index=False)
synthpop_results.to_csv("synthpop_results_detailed.csv", index=False)
synthesizer_results.to_csv("synthesizer_results_detailed.csv", index=False)

