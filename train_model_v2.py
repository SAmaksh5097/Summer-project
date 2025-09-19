import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

BIKE_COMPANIES = ['Honda', 'Royal Enfield', 'TVS', 'Ola Electric', 'Bajaj', 'Ather']

def clean_data(df):
    """A robust function to clean and standardize the dataset."""
    print(f"Initial rows: {len(df)}")
    
    # --- Clean numerical columns ---
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['kms_driven'] = df['kms_driven'].astype(str).str.replace(',', '').str.replace('kms', '').str.strip()
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Drop rows where essential data is missing
    df.dropna(subset=['year', 'kms_driven', 'Price'], inplace=True)
    df['year'] = df['year'].astype(int)
    df['kms_driven'] = df['kms_driven'].astype(int)

    # Standardize the 'Price' column to ensure all values are in Lakhs
    price_threshold = 1000  # Any price over 1000 is assumed to be in Rupees, not Lakhs
    df['Price'] = df['Price'].apply(lambda x: x / 100000 if x > price_threshold else x)
    print(f"Prices standardized. Max price in dataset is now: {df['Price'].max():.2f} Lakhs")

    print(f"Rows after cleaning essential columns: {len(df)}")
    
    # Clean and fill all other feature columns
    cat_features = ['company', 'fuel_type', 'city', 'seller_type', 'name']
    for col in cat_features:
        df[col] = df[col].fillna('Unknown')
        df[col] = df[col].astype(str).str.strip()

    df.fillna({
        'battery_capacity_kwh': 0, 'range_km': 0, 'engine_cc': 0,
        'has_sunroof': False, 'has_adas': False
    }, inplace=True)
    
    # Create derived features
    df['vehicle_type'] = np.where(df['company'].isin(BIKE_COMPANIES), 'Two-Wheeler', 'Car')
    current_year = datetime.now().year
    df['age'] = current_year - df['year']
    
    return df

class VehiclePricePredictor:
    def __init__(self):
        self.pipeline = None

    def train_model(self, df):
        X = df.drop('Price', axis=1)
        y = df['Price']
        
        numerical_features = ['year', 'kms_driven', 'age', 'battery_capacity_kwh', 'range_km', 'engine_cc']
        categorical_features = ['company', 'fuel_type', 'city', 'seller_type', 'vehicle_type', 'name']
        boolean_features = ['has_sunroof', 'has_adas']

        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), ('bool', 'passthrough', boolean_features)], remainder='drop')
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("\n🤖 Training the RandomForestRegressor on the final, standardized dataset...")
        self.pipeline.fit(X_train, y_train)
        
        # --- START: NEW EVALUATION METRICS ---
        # Generate predictions on both training and test data
        y_train_pred = self.pipeline.predict(X_train)
        y_test_pred = self.pipeline.predict(X_test)

        # Calculate metrics
        r2_train_score = r2_score(y_train, y_train_pred)
        r2_test_score = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        
        print("\n" + "="*50)
        print("FINAL MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"R² Score (Training Set): {r2_train_score:.4f}")
        print(f"R² Score (Test Set):     {r2_test_score:.4f}")
        print("-" * 50)
        print(f"Mean Absolute Error (MAE):   {mae:.4f} Lakhs")
        print(f"Mean Squared Error (MSE):    {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} Lakhs")
        print("="*50)
        # --- END: NEW EVALUATION METRICS ---

    def save_model(self, filepath='model/vehicle_model.pkl'):
        with open(filepath, 'wb') as f: pickle.dump(self.pipeline, f)
        print(f"\n✅ Final model pipeline saved to {filepath}")

if __name__ == "__main__":
    print("🚗 Final Model Training on Enriched Dataset")
    print("="*50)
    try:
        df = pd.read_excel('dataset_enriched.xlsx')
        print(f"💾 Successfully loaded 'dataset_enriched.xlsx'.")
    except FileNotFoundError:
        print("❌ ERROR: 'dataset_enriched.xlsx' not found.")
        exit()
    final_df = clean_data(df.copy())
    predictor = VehiclePricePredictor()
    predictor.train_model(final_df)
    predictor.save_model()
    print("\n✅ Final model training complete!")