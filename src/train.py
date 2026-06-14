import os
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from huggingface_hub import HfApi

def run_training():
    try:
        dataset = load_dataset('shahdadpuri/superkart-sales-dataset')
        train_df = pd.DataFrame(dataset['train'])
        test_df  = pd.DataFrame(dataset['test'])
        print('✅ Dataset loaded from Hugging Face Hub.')
    except Exception as e:
        print(f'ℹ️ HF load failed ({e}), falling back to local CSV.')
        train_df = pd.read_csv('data/train.csv')
        test_df  = pd.read_csv('data/test.csv')

    X_train = train_df.drop(columns=['Product_Store_Sales_Total'])
    y_train = train_df['Product_Store_Sales_Total']
    X_test  = test_df.drop(columns=['Product_Store_Sales_Total'])
    y_test  = test_df['Product_Store_Sales_Total']

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f'   Numeric features  : {num_cols}')
    print(f'   Categorical features: {cat_cols}')

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(),              num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    pipeline = Pipeline([
        ('prep',  preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    param_grid = {'model__n_estimators': [100], 'model__max_depth': [15]}
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    preds = search.best_estimator_.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    print('=' * 50)
    print('   PRODUCTION MODEL PERFORMANCE VALIDATION')
    print('=' * 50)
    print(f' Best Params : {search.best_params_}')
    print(f' RMSE        : {rmse:.2f}')
    print(f' R² Score    : {r2:.4f}')
    print('=' * 50)

    joblib.dump(search.best_estimator_, 'best_sales_model.joblib')
    print('✅ Model saved: best_sales_model.joblib')

    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        api = HfApi()
        api.create_repo(repo_id='shahdadpuri/superkart-sales-model',
                        repo_type='model', exist_ok=True, token=hf_token)
        api.upload_file(path_or_fileobj='best_sales_model.joblib',
                        path_in_repo='best_sales_model.joblib',
                        repo_id='shahdadpuri/superkart-sales-model',
                        repo_type='model', token=hf_token)
        print('✅ Model uploaded to HF Model Hub: shahdadpuri/superkart-sales-model')

if __name__ == '__main__':
    run_training()
