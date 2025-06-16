import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os



def run_preprocessing_pipeline(RAW_DATA_PATH,PROCESSED_DATA_DIR):
    print("Starting data preprocessing automation...")

    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Load raw data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print("Raw dataset loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}. Exiting.")
        return # Exit if raw data not found for automation
 
    # 2. Drop Duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Duplicates removed: {initial_rows - df.shape[0]} rows.")

    # 3. Feature Engineering (BHK & total_sqft conversion)
    df['bedrooms'] = df['size'].str.extract('(\d+)').astype(float)
    df.drop('size', axis=1, inplace=True, errors='ignore')

    def convert_sqft_for_automation(x): # Robust converter
        if isinstance(x, str):
            tokens = x.split('-')
            if len(tokens) == 2:
                try: return (float(tokens[0]) + float(tokens[1])) / 2
                except ValueError: return np.nan
            try: return float(x)
            except ValueError: return np.nan
        return float(x) if pd.api.types.is_numeric_dtype(x) else np.nan

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_for_automation)
    
    # 4. Handle Missing Values (Imputation for numerical features)
    # Critical NaNs (price, total_sqft, bath) are dropped first
    df.dropna(subset=['price', 'total_sqft', 'bath'], inplace=True)
    
    # Impute remaining numerical features with median
    num_cols_for_imputation = ['total_sqft', 'bath', 'balcony', 'bedrooms']
    for col in num_cols_for_imputation:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
    
    # Impute categorical features with mode (e.g., society, location if NaNs remain)
    cat_cols_for_imputation = ['area_type', 'availability', 'location', 'society']
    for col in cat_cols_for_imputation:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True) # Use mode for categorical NaNs

    # 5. Create price_per_sqft feature (after total_sqft and price are clean)
    if 'price' in df.columns and 'total_sqft' in df.columns and (df['total_sqft'] != 0).any():
        df['price_per_sqft'] = df['price'] / df['total_sqft']
    else:
        print("Warning: Skipping price_per_sqft feature due to missing 'price' or 'total_sqft' or zero 'total_sqft'.")

    # 6. Outlier Handling (total_sqft/bedrooms)
    if 'total_sqft' in df.columns and 'bedrooms' in df.columns and (df['bedrooms'] != 0).any():
        df = df[~(df['total_sqft'] / df['bedrooms'] < 300)].copy()
    else:
        print("Warning: Skipping total_sqft/bedrooms outlier removal.")

    # 7. Outlier Handling (price_per_sqft quantiles)
    if 'price_per_sqft' in df.columns:
        lower_limit_pps = df['price_per_sqft'].quantile(0.01)
        upper_limit_pps = df['price_per_sqft'].quantile(0.99)
        df = df[(df['price_per_sqft'] >= lower_limit_pps) & (df['price_per_sqft'] <= upper_limit_pps)].copy()
    else:
        print("Warning: Skipping price_per_sqft quantile outlier removal.")

    # 8. Outlier Handling (price_per_sqft per location)
    if 'price_per_sqft' in df.columns and 'location' in df.columns:
        def remove_pps_outliers_per_location_auto(df_temp_outlier):
            df_out = pd.DataFrame()
            for key, subdf in df_temp_outlier.groupby('location'):
                if not subdf.empty:
                    m = np.mean(subdf.price_per_sqft)
                    st = np.std(subdf.price_per_sqft)
                    reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
                    df_out = pd.concat([df_out, reduced_df], ignore_index=True)
            return df_out
        df = remove_pps_outliers_per_location_auto(df)
    else:
        print("Warning: Skipping price_per_sqft per location outlier removal.")


    # 9. Dimensionality Reduction for Location and Society
    if 'location' in df.columns:
        df['location'].fillna('Unknown', inplace=True)
        location_stats = df['location'].value_counts(ascending=False)
        location_less_than_10 = location_stats[location_stats <= 10]
        df['location'] = df['location'].apply(lambda x: 'other' if x in location_less_than_10 else x)

    if 'society' in df.columns:
        df['society'].fillna('Unknown', inplace=True)
        top_societies = df['society'].value_counts().head(10).index
        df['society'] = df['society'].apply(lambda x: x if x in top_societies else 'Other')

    # 10. Prepare Features (X) and Target (y)
    # At this point, df has all processed features and the 'price' target.
    # Categorical columns are still present for OneHotEncoding by the pipeline.
    X = df.drop(['price', 'price_per_sqft'], axis=1, errors='ignore') # Ensure 'price_per_sqft' is not in final features
    y = df['price']

    # 11. Train-Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 12. Create and Apply Preprocessing Pipeline (Numerical & Categorical)
    # Identify numerical and categorical features for the ColumnTransformer
    # These should be the column names *before* OneHotEncoding by the pipeline
    num_features_pipeline = ['total_sqft', 'bath', 'balcony', 'bedrooms', 'price_per_sqft']
    cat_features_pipeline = ['area_type', 'availability', 'location', 'society']

    # Filter to only include columns that are actually in X_train_raw
    num_features_pipeline = [col for col in num_features_pipeline if col in X_train_raw.columns]
    cat_features_pipeline = [col for col in cat_features_pipeline if col in X_train_raw.columns]

    # Numerical features pipeline (impute NaNs, scale)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features pipeline (impute NaNs, one-hot encode)
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Column transformer: applies different pipelines to different column sets
    preprocessor = ColumnTransformer(
        [
            ('num', num_pipeline, num_features_pipeline),
            ('cat', cat_pipeline, cat_features_pipeline)
        ],
        remainder='passthrough'
    )

    # Apply Preprocessing: Fit on training data and transform both train/test sets
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    print("Preprocessing completed! (Pipeline applied)")
    print(f"Training set shape after final preprocessing: {X_train_processed.shape}")
    print(f"Test set shape after final preprocessing: {X_test_processed.shape}")

    # 13. Save Processed Data to CSV files (Crucial for output)

    # Ensure PROCESSED_DATA_DIR exists before saving
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    X_train_final_df = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
    X_test_final_df = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())

    X_train_final_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    X_test_final_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
    
    print(f"\nProcessed data saved to {PROCESSED_DATA_DIR}/")
    print(f"X_train_processed shape: {X_train_processed.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_processed shape: {X_test_processed.shape}, y_test shape: {y_test.shape}")

    print("\nAutomation script execution complete.")

if __name__ == "__main__":
    
    # --- Configuration ---
    RAW_DATA_PATH = 'Housing_raw/Bengaluru_House_Data.csv'
    PROCESSED_DATA_DIR = 'preprocessing/housing_preprocessing' # Output directory for processed data

    
    run_preprocessing_pipeline(RAW_DATA_PATH,PROCESSED_DATA_DIR) # Call the main function to run all steps
