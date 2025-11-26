import pandas as pd
import os

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data')
raw_data_path = os.path.join(data_dir, 'raw', 'student_performance.csv')
processed_data_path = os.path.join(data_dir, 'processed', 'student_performance_cleaned.csv')

def preprocess():

    print("Loading raw data...")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")
    
    df = pd.read_csv(raw_data_path)
    print("Raw data loaded successfully.")


    print("Preprocessing data...")

    # Data Cleaning Steps
    # Drop columns that are not useful for prediction
    cols_to_drop = ['StudentID', 'GradeClass']  # GradeClass is leakage
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"Dropped columns: {cols_to_drop}")

    # Quality checks
    if 'GPA' not in df_cleaned.columns:
        raise ValueError("Target column 'GPA' is missing from the dataset after preprocessing.")    
    
    # Save cleaned data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_cleaned.to_csv(processed_data_path, index=False)
    print(f"Preprocessed data saved to {processed_data_path}")

    print("Preprocessing completed.")
    print(f"Cleaned data shape: {df_cleaned.shape}")
    print(f"Columns in cleaned data: {df_cleaned.columns.tolist()}")

if __name__ == "__main__":
    preprocess()