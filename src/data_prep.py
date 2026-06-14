import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def run_pipeline():
    df = pd.read_csv('SuperKart.csv')
    print(f'📊 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns')

    # Fix encoding inconsistency
    df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({'reg': 'Regular'})

    # Drop ID columns — not predictive features
    df_cleaned = df.drop(columns=['Product_Id', 'Store_Id'], errors='ignore')

    train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

    # Reset index to avoid index_level_0 column issue in HF datasets
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv',   index=False)
    print(f'✅ Train: {len(train_df)} rows | Test: {len(test_df)} rows — saved to data/')

    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        hf_dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df, preserve_index=False),
            'test':  Dataset.from_pandas(test_df,  preserve_index=False)
        })
        hf_dataset.push_to_hub('shahdadpuri/superkart-sales-dataset', token=hf_token)
        print('✅ Dataset pushed to Hugging Face Hub.')

if __name__ == '__main__':
    run_pipeline()
